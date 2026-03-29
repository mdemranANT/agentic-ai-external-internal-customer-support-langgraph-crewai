"""
Embedding and ChromaDB ingestion.
Loads the department FAQ JSON files, embeds them with OpenAI embeddings,
and stores everything in a local ChromaDB instance with department metadata
so we can do filtered retrieval later.

Usage:  python vector_store.py
"""
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    DATA_DIR,
    RAG_TOP_K,
    RAG_SCORE_THRESHOLD,
)


def load_all_documents() -> list[Document]:
    """Reads every *_faq.json in the data folder and wraps each QA pair as a Document."""
    data_path = Path(DATA_DIR)
    documents = []

    for faq_file in sorted(data_path.glob("*_faq.json")):
        with open(faq_file, "r", encoding="utf-8") as f:
            doc = json.load(f)

        dept_key = doc["department_key"]
        dept_name = doc["department"]
        audience = doc["audience"]

        for entry in doc["entries"]:
            page_content = f"Question: {entry['question']}\nAnswer: {entry['answer']}"
            metadata = {
                "department_key": dept_key,
                "department_name": dept_name,
                "audience": audience,
                "tags": ", ".join(entry.get("tags", [])),
                "source": faq_file.name,
            }
            documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


def create_vector_store(documents: list[Document] | None = None) -> Chroma:
    """Build a new ChromaDB collection (if documents given) or load an existing one."""
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )

    if documents:
        print(f"  Embedding {len(documents)} documents into ChromaDB...")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=CHROMA_COLLECTION_NAME,
            collection_metadata={"hnsw:space": "cosine"},
            persist_directory=CHROMA_PERSIST_DIR,
        )
        print(f"  [OK] Vector store created at {CHROMA_PERSIST_DIR}")
    else:
        vector_store = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        print(f"  [OK] Loaded existing vector store from {CHROMA_PERSIST_DIR}")

    return vector_store


def get_retriever(vector_store: Chroma, department_key: str | None = None):
    """Returns a LangChain retriever, optionally filtered to a single department."""
    search_kwargs = {
        "k": RAG_TOP_K,
        "score_threshold": RAG_SCORE_THRESHOLD,
    }
    if department_key:
        search_kwargs["filter"] = {"department_key": department_key}

    return vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs=search_kwargs,
    )


def retrieve_context(
    vector_store: Chroma,
    query: str,
    department_key: str | None = None,
    fallback_to_global: bool = True,
) -> tuple[str, list]:
    """
    Pulls relevant docs from ChromaDB and returns (context_string, raw_docs).

    If nothing comes back for the given department and fallback_to_global is
    True, we retry without the department filter so the user at least gets
    a partial answer.
    """
    retriever = get_retriever(vector_store, department_key)
    docs = retriever.invoke(query)

    # If department-specific search came up empty, try without the filter
    used_fallback = False
    if not docs and fallback_to_global and department_key:
        retriever = get_retriever(vector_store, department_key=None)
        docs = retriever.invoke(query)
        used_fallback = bool(docs)

    if not docs:
        return ("", [])

    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    # If we used the global fallback, prepend a note about cross-department source
    if used_fallback and docs:
        source_depts = set(d.metadata.get("department_key", "") for d in docs)
        source_depts.discard(department_key)
        if source_depts:
            dept_names = ", ".join(sorted(source_depts))
            context = (
                f"[NOTE: No results found in {department_key} knowledge base. "
                f"The following context comes from other departments: {dept_names}. "
                f"You should suggest transferring the user to the correct department.]\n\n"
                + context
            )

    return (context, docs)


# ---------- CLI entry point: ingest everything ----------
if __name__ == "__main__":
    print("=" * 60)
    print("Shop4You  --  Vector Store Ingestion")
    print("=" * 60)

    docs = load_all_documents()
    print(f"  Loaded {len(docs)} documents from {DATA_DIR}/")

    vs = create_vector_store(docs)

    # Quick test
    print("\n  Quick retrieval test:")
    test_query = "How do I return an item?"
    context, docs = retrieve_context(vs, test_query, "orders_returns")
    if context:
        print(f"  [OK] Retrieved {len(docs)} doc(s) for: '{test_query}'")
        print(f"    Preview: {context[:150]}...")
    else:
        print(f"  [FAIL] No results for test query")

    print("\n" + "=" * 60)
    print("Done. Vector store ready.")
