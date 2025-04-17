import os
import subprocess
import re
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import GitLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import shutil
import stat
INCLUDE_EXTS = ('.py','.md','.ipynb','.txt','.json','.yml','.yaml')
CLONE_BASE = "./cloned_repos"
CHROMA_PATH = "./chroma_store"
TOP_K = 10              # how many to fetch before filtering
DIST_THRESH = 0.6       # max cosine distance (smaller => more similar)

def _rm_error_handler(func, path, exc_info):
    """
    If removal fails, attempt to chmod the file to writable and retry.
    """
    # Is it a permission error?
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWRITE)
        func(path)  # retry

def load_github_repo(repo_url, branch="main", file_types=None):
    """
    Clones the GitHub repo to ./cloned_repos/<repo_name>,
    using GitLoader to load files, and safely wipes any old clone.
    """
    repo_name = repo_url.rstrip("/").split("/")[-1]
    repo_path = os.path.join("cloned_repos", repo_name)

    # Remove old clone safely on all platforms
    if os.path.isdir(repo_path):
        print(f"🧹 Removing existing directory {repo_path}")
        shutil.rmtree(repo_path, onerror=_rm_error_handler)

    # Clone fresh
    print(f"📦 Cloning {repo_url}@{branch} into {repo_path} …")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", branch, repo_url, repo_path],
        check=True
    )

    # Load with GitLoader
    loader = GitLoader(
        repo_path=repo_path,
        branch=branch,
        file_filter=lambda p: any(p.endswith(ext) for ext in file_types)
    )
    documents = loader.load()
    return documents
# ────────────────────────────────────────────────────────────────
# 2️⃣ Text Splitting
# ────────────────────────────────────────────────────────────────

def split_text(text: str, chunk_size=1500, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n","\n"," ",""]
    )
    # drop empty
    return [c for c in splitter.split_text(text) if c.strip()]

# ────────────────────────────────────────────────────────────────
# 3️⃣ Keyword Snippets (Exact‑Match Reinforcement)
# ────────────────────────────────────────────────────────────────

def keyword_snippets(repo_path: str, query: str, max_snips=5):
    """Greps the local clone for small windows around query keywords."""
    snippets = []
    pattern = re.compile(rf".{{0,50}}{re.escape(query)}.{{0,50}}", re.IGNORECASE)
    for root,_,files in os.walk(repo_path):
        for fn in files:
            if fn.endswith(INCLUDE_EXTS):
                full = os.path.join(root,fn)
                text = open(full, errors="ignore").read()
                for m in pattern.finditer(text):
                    snippets.append(f"...{m.group()}...")
                    if len(snippets)>=max_snips:
                        return snippets
    return snippets

# ────────────────────────────────────────────────────────────────
# 4️⃣ ChromaDB + Embeddings + LLM Setup
# ────────────────────────────────────────────────────────────────

# Persistent Chroma client
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Nomic embeddings via HF wrapper
emb_model = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code":True}
)

# Get or create our collection
collection = client.get_or_create_collection(name="github_repo_chat")

# Gemini‑2 LLM
gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key='AIzaSyCGbuEtVoL9-b1K7O1Tn7qS94Vf9bzU9as'
)

# ────────────────────────────────────────────────────────────────
# 5️⃣ Indexing: Split → Embed → Add to ChromaDB
# ────────────────────────────────────────────────────────────────

def add_chunks_to_chroma(documents, repo_path):
    """
    Takes a list of LangChain Document objects, splits them,
    embeds them in batches, and adds to ChromaDB.
    """
    for doc in documents:
        src = doc.metadata.get("source", "unknown")
        chunks = split_text(doc.page_content)
        if not chunks:
            continue
        print(f"Indexing {src} → {len(chunks)} chunks")

        ids       = [f"{src}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"file":src} for _ in chunks]
        embeddings = emb_model.embed_documents(chunks)
        if not embeddings:
            print(f"⚠️ no embeddings for {src}, skipping")
            continue

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    print("✅ All docs indexed.")

# ────────────────────────────────────────────────────────────────
# 6️⃣ Retrieval with Filtering
# ────────────────────────────────────────────────────────────────

def retrieve(query: str, repo_path: str):
    # Embed the user query
    q_emb = emb_model.embed_query(query)

    # Simple vector search: top‑K results with distances
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=TOP_K,
        include=["documents","metadatas","distances"]
    )

    docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]

    # Filter out anything with distance ≥ DIST_THRESH
    filtered = [(d, m) for d, m, dist in zip(docs, metas, dists) if dist < DIST_THRESH]
    return filtered

# ────────────────────────────────────────────────────────────────
# 7️⃣ Strict Prompt Template
# ────────────────────────────────────────────────────────────────

PROMPT = """You are a code‑reading assistant.  
ONLY use the CONTEXT below to answer.  
If the answer is not in the context, reply exactly “I don’t know.”

CONTEXT:
{context}

QUESTION:
{question}

Answer concisely, citing file names when relevant.
"""

# ────────────────────────────────────────────────────────────────
# 8️⃣ Main Chatbot Loop
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    repo_url = input("🔗 GitHub Repo URL: ").strip()
    branch   = input("🔀 Branch (default main): ").strip() or "main"
    print(f"⏳ Cloning & loading `{repo_url}` @ {branch}…")

    docs = load_github_repo(repo_url, branch, file_types=INCLUDE_EXTS)
    repo_name = repo_url.rstrip("/").split("/")[-1]
    repo_path = os.path.join(CLONE_BASE, repo_name)
    print(f"✅ Loaded {len(docs)} files from repo.")

    add_chunks_to_chroma(docs, repo_path)

    print("\n💬 Chatbot ready — type ‘exit’ to quit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower()=="exit":
            print("👋 Bye!"); break

        # 1) Retriever
        filtered = retrieve(q, repo_path)
        if not filtered:
            print("❌ No relevant content found. Try rephrasing.")
            continue

        top_docs = [d for d,_ in filtered]
        context   = "\n\n".join(top_docs)

        # 2) Keyword reinforcement
        snips = keyword_snippets(repo_path, q)
        if snips:
            context += "\n\nKEYWORD SNIPPETS:\n" + "\n".join(snips)

        # 3) Prompt & LLM call
        prompt = PROMPT.format(context=context, question=q)
        ans = gemini.invoke(prompt).content

        # 4) Display
        print(f"\n🤖 Answer:\n{ans}\n")
        print(".sources:")
        for _,meta in filtered:
            print("-", meta["file"])
        print("-"*60+"\n")
