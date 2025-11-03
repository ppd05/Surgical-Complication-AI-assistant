#vectorstores

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def build_faiss_retriever(docs):
    #Build FAISS retriever using free Hugging Face embeddings.#
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embedding=embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 3})