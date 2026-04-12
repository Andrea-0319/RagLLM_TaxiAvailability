from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from documents import documents

# embedding
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# crea DB
db = FAISS.from_texts(documents, embedding)


def retrieve_context(user_input, k=3):
    results = db.similarity_search(user_input, k=k)
    return "\n".join([r.page_content for r in results])