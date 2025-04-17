import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import GitLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import subprocess
from langchain_community.document_loaders import GitLoader

import shutil
import stat

def on_rm_error(func, path, exc_info):
    """Error handler for removing read-only files on Windows."""
    os.chmod(path, stat.S_IWRITE)  # Remove read-only flag
    func(path)

def load_github_repo(repo_url, branch="main", file_types=None):
    # Prepare local path
    repo_name = repo_url.split('/')[-1]
    repo_path = f"./cloned_repos/{repo_name}"

    # If repo already exists — remove it first (to avoid conflicts)
    if os.path.exists(repo_path):
        print(f"🧹 Removing existing repo at {repo_path}")
        shutil.rmtree(repo_path, onerror=on_rm_error)

    # Clone the repo
    subprocess.run(["git", "clone", "--branch", branch, repo_url, repo_path], check=True)

    # Load files with GitLoader
    loader = GitLoader(
        repo_path=repo_path,
        branch=branch,
        file_filter=lambda file_path: any(file_path.endswith(ext) for ext in file_types)
    )
    documents = loader.load()
    return documents

# 2️⃣ Split text into chunks
def split_text(text, chunk_size=1500, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

# 3️⃣ Initialize ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path="./chroma_store")

# 4️⃣ Initialize Embeddings and LLM
embeddings_model = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code": True}
)

gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key='AIzaSyCGbuEtVoL9-b1K7O1Tn7qS94Vf9bzU9as')


# 7️⃣ Query Chroma collection
def query_chroma(query, top_k=5):
    query_embedding = embeddings_model.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results

def init_chroma_client(repo_name):
    store_path = f"./chroma_store/{repo_name}"
    os.makedirs(store_path, exist_ok=True)  # make sure the folder exists
    return chromadb.PersistentClient(path=store_path)

def add_chunks_to_chroma(documents, collection):
    for doc in documents:
        file_path = doc.metadata.get('source', 'unknown')
        chunks = split_text(doc.page_content)

        if not chunks:
            print(f"⚠️ No chunks found for {file_path}, skipping.")
            continue

        print(f"📄 {file_path} — {len(chunks)} chunks")

        ids = [f"{file_path}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"file": file_path} for _ in chunks]

        embeddings = embeddings_model.embed_documents(chunks)

        if not embeddings:
            print(f"❌ Failed to generate embeddings for {file_path}, skipping.")
            continue

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ Added {len(chunks)} chunks from {file_path}")

# 8️⃣ Main execution
if __name__ == "__main__":
    repo_url = input("Enter GitHub repository URL: ")
    branch = input("Enter branch (default is 'main'): ") or "main"

    repo_name = repo_url.split('/')[-1]
    print(f"📥 Loading repository: {repo_url} (branch: {branch})")
    include_extensions = ('.py', '.md', '.ipynb', '.txt', '.json', '.yml', '.yaml')
    documents = load_github_repo(repo_url, branch, file_types=include_extensions)
    print(f"✅ Loaded {len(documents)} documents")

    # 🔸 Initialize a **separate chroma store for each repo**
    chroma_client = init_chroma_client(repo_name)

    # 🔸 Create a collection specific to this repo
    collection = chroma_client.get_or_create_collection(name=f"github_repo_chat_{repo_name}")

    add_chunks_to_chroma(documents, collection)
    print("📚 Repository indexed into ChromaDB!")

    print("\n💬 Repo Chatbot with Gemini is ready! (type 'exit' to quit)\n")

    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            print("👋 Exiting chat. Goodbye!")
            break

        # Retrieve relevant code context
        query_embedding = embeddings_model.embed_query(user_query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )
        relevant_chunks = results['documents'][0]
        sources = results['metadatas'][0]

        if not relevant_chunks:
            print("❌ No relevant code found in repository")
            continue

        context = "\n\n".join(relevant_chunks)
        prompt = f"""Analyze the following code context from a GitHub repository and answer the user's question.
        Focus on technical implementation details and patterns found in the codebase.

        Context:
        {context}

        Question: {user_query}

        Provide a detailed answer with code references where applicable.
        """

        response = gemini.invoke(prompt)

        print(f"\n🤖 Gemini Analysis:\n{response.content}")
        print("\n📁 Source Files:")
        for src in {s['file'] for s in sources}:
            print(f"- {src}")
        print("-" * 50 + "\n")

