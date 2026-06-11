import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core import documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcripts"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"}
        
        )

def create_vector_store(transcript : str)-> Chroma:
    # Split the transcript into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(transcript)
    
    docs = [
        Document(
            page_content=chunk,
            metadata={"source": f"chunk_{i}"}
        )
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )
    return vector_store

def load_vector_store() -> Chroma:
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(f"Vector store directory '{CHROMA_DIR}' does not exist.")
    
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    return vector_store

def get_retriever(vector_store: Chroma):
    retriever = vector_store.as_retriever(
        search_type = 'similarity',
          search_kwargs={"k": 4}
          )
    return retriever