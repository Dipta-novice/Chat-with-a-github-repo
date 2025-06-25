import os
import textwrap
import tempfile
import uuid
import re
import threading
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import GithubFileLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableMap
from langchain_google_genai import ChatGoogleGenerativeAI


# === Timeout Handler ===
class TimeoutException(Exception):
    pass

def timeout_handler():
    raise TimeoutException("The operation timed out.")

# === Prettify printing function ===
def format_response(content: str) -> str:
    content = content.replace("\\n", "\n").replace("\n\n", "\n\n")
    formatted_lines = []
    for line in content.splitlines():
        if line.strip().startswith("###"):
            formatted_lines.append(f"\n\033[1m{line.strip()}\033[0m")
        elif line.strip().startswith("1.") or line.strip().startswith("-"):
            formatted_lines.append(f"  {line.strip()}")
        else:
            wrapped = textwrap.fill(line.strip(), width=100)
            formatted_lines.append(wrapped)
    return "\n".join(formatted_lines)


# === GitHub Repo Input ===
repo_url = input("Enter GitHub repo URL:\n> ")
branch_name = input("Enter branch name (default is 'main'):\n> ").strip() or "main"
match = re.search(r'github\.com/([^/]+/[^/]+)', repo_url)
repo_path = match.group(1) if match else None

# === GitHub Token (Hardcoded or from env) ===
access_token = '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'
if not access_token:
    raise ValueError("GitHub token not found. Please set it.")


# === Timeout Configurations ===

# === Gemini API Timeout ===
def gemini_with_timeout(prompt: str, timeout=60):
    timer = threading.Timer(timeout, timeout_handler)
    try:
        timer.start()
        response = gemini.invoke(prompt)  # Call to Gemini API
        timer.cancel()  # Cancel the timer if successful
        return response
    except TimeoutException:
        return "❌ Timeout: The Gemini API took too long to respond."
    except Exception as e:
        timer.cancel()
        raise e


# === Clone and Load GitHub Files ===
unique_clone_path = os.path.join(tempfile.gettempdir(), f"gh_clone_{uuid.uuid4().hex}")
loader = GithubFileLoader(
    repo=repo_path,
    access_token=access_token,
    branch=branch_name,
    clone_path=unique_clone_path,
    file_filter=lambda file_path: file_path.endswith(('.py', '.md', '.ipynb', '.txt', '.json', '.yml', '.yaml'))
)

docs = loader.load()


# === Parse Documents ===
# === Multi-language Parsing & Splitting ===

# Define parsers for supported languages
parsers = {
    ".py": LanguageParser(language=Language.PYTHON, parser_threshold=500),
    ".js": LanguageParser(language=Language.JS, parser_threshold=500),
    ".ts": LanguageParser(language=Language.JS, parser_threshold=500),
    ".jsx": LanguageParser(language=Language.JS, parser_threshold=500),
    ".mjs": LanguageParser(language=Language.JS, parser_threshold=500),
    ".cjs": LanguageParser(language=Language.JS, parser_threshold=500),
    ".ejs": LanguageParser(language=Language.JS, parser_threshold=500),
    ".pug": LanguageParser(language=Language.JS, parser_threshold=500),
    ".php": LanguageParser(language=Language.PHP, parser_threshold=500),
    ".blade": LanguageParser(language=Language.PHP, parser_threshold=500),
}

# Store parsed and split documents
texts = []

# Iterate through each document and process by language
for doc in docs:
    source = doc.metadata.get("source", "")
    ext = os.path.splitext(source)[-1].lower()
    parser = parsers.get(ext)

    if parser:
        blob = Blob.from_data(doc.page_content, path=source)
        try:
            parsed = parser.parse(blob)
            # Use matching splitter based on language
            lang = parser.language
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang,
                chunk_size=1000,
                chunk_overlap=200
            )
            texts.extend(splitter.split_documents(parsed))
        except Exception as e:
            print(f"❌ Failed to parse {source}: {e}")
    else:
        # Default: split raw document if unsupported
        default_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=200)
        texts.extend(default_splitter.split_documents([doc]))

# === Embeddings ===
embedding = OllamaEmbeddings(
    base_url="http://192.168.0.214:11434",
    model="granite-embedding:278m"
)

# === Vector Store ===
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embedding,
    persist_directory="chroma_db"
)

retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

# === LLMs ===
#ollama = ChatOllama(
    base_url="http://192.168.0.214:11434",
    model="qwen2.5-coder:14b",
    temperature=0.4
)

gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key='AIzaSyCGbuEtVoL9-b1K7O1Tn7qS94Vf9bzU9as'
)

# === Memory ===
memory = ConversationBufferMemory(return_messages=True)

# === Prompts ===
qa_prompt = PromptTemplate.from_template(
    """You are a helpful coding assistant that answers questions about the GitHub repository.

Previous conversation:
{history}

User question: {input}
"""
)

classifier_prompt = PromptTemplate.from_template(
    "You are a codebase analysis assistant. Determine if the following question is broad or specific.\n"
    "Broad questions typically involve general overviews, architecture, vulnerabilities, or design principles.\n"
    "Specific questions involve exact functions, files, or code snippets.\n\n"
    "Respond with only 'yes' if the question is broad, or 'no' if it is specific.\n\n"
    "Question: {question}"
)

system_message = """
You are a senior software engineer and code reviewer.

Your job is to respond to the user's question **only by analyzing the codebase**.

DO NOT:
- Describe what the code does (unless the user explicitly asks for that).
- List or explain which libraries are used.
- Summarize the codebase.
- Return boilerplate documentation-style responses.

DO:
- Critically analyze the codebase based on the user's question.
- Identify issues with performance, security, design, testing, or maintainability.
- Offer actionable suggestions in bullet points or a numbered list.
- Be specific and technical in your recommendations.
- If relevant, suggest better practices or libraries.

Answer clearly, concisely, and professionally.
"""

full_repo_prompt = PromptTemplate.from_template(
    """<|system|>
{system_message}
<|end|>

<|user|>
{user_question}

Full repository content:
{repo_content}
<|end|>
"""
)

# === Chain with memory awareness ===
chain: Runnable = (
    RunnableMap({
        "input": lambda x: x["input"],
        "history": lambda x: memory.load_memory_variables(x)["history"]
    })
    | qa_prompt
    | ollama
    | StrOutputParser()
)

# === Classification Function ===
def is_broad_question(question: str) -> bool:
    prompt = classifier_prompt.format(question=question)
    response = ollama.invoke(prompt)
    return 'yes' in response.content.lower()

# === Chat Handler ===
from langchain_core.messages import HumanMessage, SystemMessage

def chat_with_repo(user_input: str):
    if is_broad_question(user_input):
        print("Broad question detected, analyzing full repo...")

        full_content = "\n".join([doc.page_content for doc in texts])

        prompt_text = full_repo_prompt.format(
            user_question=user_input,
            repo_content=full_content,
            system_message=system_message
        )

        # Ollama expects a list of messages, not a string
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=prompt_text)
        ]

        try:
            response = ollama.invoke(messages)
        except TimeoutException:
            response = "❌ Timeout: The Ollama API took too long to respond."
    else:
        try:
            response = chain.invoke({"input": user_input})
        except TimeoutException:
            response = "❌ Timeout: The chain processing took too long to complete."

    memory.save_context({"input": user_input}, {"output": response.content if hasattr(response, "content") else response})

    pretty_output = format_response(response.content if hasattr(response, "content") else response)
    print("\nAnswer:\n", pretty_output)

# === REPL Loop ===
if __name__ == "__main__":
    while True:
        user_input = input("\nAsk a question about the repo (type 'exit' to quit):\n> ")
        if user_input.lower() == "exit":
            break
        chat_with_repo(user_input)
