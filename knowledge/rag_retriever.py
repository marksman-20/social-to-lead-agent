import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

KB_PATH = os.path.join(os.path.dirname(__file__), "autostream_kb.md")
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "faiss_index")

def get_embeddings():
    """Return local HuggingFace embeddings."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_kb():
    """Builds and saves the FAISS index from the markdown knowledge base."""
    print("Loading Knowledge Base...")
    loader = TextLoader(KB_PATH)
    docs = loader.load()

    print("Splitting document...")
    # Markdown-friendly splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n## ", "\n### ", "\n- ", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)

    print(f"Creating FAISS index with {len(splits)} chunks using sentence-transformers...")
    vectorstore = FAISS.from_documents(documents=splits, embedding=get_embeddings())
    
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"Index saved to {FAISS_INDEX_PATH}")

def get_retriever():
    """Returns a retriever interface for the stored FAISS index."""
    if not os.path.exists(FAISS_INDEX_PATH):
        print("FAISS index not found. Building it now...")
        build_kb()
        
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH, 
        get_embeddings(), 
        allow_dangerous_deserialization=True # required for local FAISS loading
    )
    # k=3 retrieves the top 3 most relevant chunks
    return vectorstore.as_retriever(search_kwargs={"k": 3})

if __name__ == "__main__":
    # If run directly as a script, just build the KB
    build_kb()
