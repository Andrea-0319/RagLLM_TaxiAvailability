from __future__ import annotations
from typing import Optional

_db = None


def retrieve_context(query: str, k: int = 3) -> str:
    global _db
    if _db is None:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from .rag_documents import documents

        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        _db = FAISS.from_texts(documents, embedding)

    results = _db.similarity_search(query, k=k)
    return "\n".join([r.page_content for r in results])