"""
IT Incident Knowledge Assistant
A very simple RAG (Retrieval-Augmented Generation) command-line app.

Pipeline:
    knowledge_base/*.txt -> TextSplitter -> Embeddings -> FAISS
    -> similarity search (top 2) -> single LLM call -> structured answer
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---------------------------------------------------------------- settings --
BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge_base"
INDEX_DIR = BASE_DIR / "faiss_index"      # cached vectors, rebuilt if missing

TOP_K = 2                                  # number of chunks retrieved
MIN_RELEVANCE = 0.25                       # below this, treat as "no match"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

EMBED_MODEL = "text-embedding-3-small"     # cheap embedding model
CHAT_MODEL = "gpt-4o-mini"                 # cheap chat model

# ------------------------------------------------------------------ prompt --
PROMPT = ChatPromptTemplate.from_template(
    """You are an IT support assistant.

Answer ONLY using the context below. Do not use outside knowledge and do not
invent commands, file paths or settings that are not in the context.

If the context does not contain enough information to solve the incident,
reply with exactly this single line and nothing else:
NOT_ENOUGH_INFORMATION

Otherwise reply in this exact format, in under 400 words:

Problem:
<one or two sentences>

Possible Cause:
<causes supported by the context only>

Troubleshooting Steps:
1. <step>
2. <step>
3. <step>

Resolution:
<recommended fix>

Verification:
<how to confirm the incident is resolved>

Context:
{context}

Incident reported by the user:
{question}"""
)


# ------------------------------------------------------------- vector store --
def build_vector_store(embeddings: OpenAIEmbeddings) -> FAISS:
    """Load the .txt knowledge base, split it, embed it and store it in FAISS."""
    if INDEX_DIR.exists():
        return FAISS.load_local(
            str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )

    if not KB_DIR.exists():
        sys.exit(f"Knowledge base folder not found: {KB_DIR}")

    loader = DirectoryLoader(
        str(KB_DIR),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    if not documents:
        sys.exit(f"No .txt files found in {KB_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(INDEX_DIR))
    print(f"Indexed {len(documents)} documents into {len(chunks)} chunks.\n")
    return store


# ---------------------------------------------------------------- retrieval --
def retrieve(store: FAISS, question: str):
    """Return the top-K chunks that are above the relevance threshold."""
    results = store.similarity_search_with_relevance_scores(question, k=TOP_K)
    return [(doc, score) for doc, score in results if score >= MIN_RELEVANCE]


def source_names(docs) -> list[str]:
    """Unique file names of the retrieved chunks, in order."""
    names = []
    for doc in docs:
        name = Path(doc.metadata.get("source", "unknown")).name
        if name not in names:
            names.append(name)
    return names


# -------------------------------------------------------------------- main --
def answer_incident(store: FAISS, llm: ChatOpenAI, question: str) -> None:
    matches = retrieve(store, question)

    if not matches:
        print("\nRetrieved documents: none")
        print(
            "\nSufficient information was not found in the knowledge base "
            "to answer this incident."
        )
        return

    docs = [doc for doc, _ in matches]
    print("\nRetrieved documents:")
    for name, (_, score) in zip(source_names(docs), matches):
        print(f"  - {name} (relevance {score:.2f})")

    context = "\n\n---\n\n".join(
        f"[{Path(d.metadata.get('source', 'unknown')).name}]\n{d.page_content}"
        for d in docs
    )

    # one single LLM call per question
    response = llm.invoke(PROMPT.format_messages(context=context, question=question))
    text = response.content.strip()

    if "NOT_ENOUGH_INFORMATION" in text:
        print(
            "\nSufficient information was not found in the knowledge base "
            "to answer this incident."
        )
        return

    print("\n" + text + "\n")


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Put it in a .env file or export it.")

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    store = build_vector_store(embeddings)

    # single question mode:  python app.py "disk is full on D drive"
    if len(sys.argv) > 1:
        answer_incident(store, llm, " ".join(sys.argv[1:]))
        return

    print("IT Incident Knowledge Assistant (type 'exit' to quit)")
    while True:
        question = input("\nDescribe the incident > ").strip()
        if question.lower() in {"exit", "quit", ""}:
            print("Bye.")
            return
        answer_incident(store, llm, question)


if __name__ == "__main__":
    main()
