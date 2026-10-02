from chunker import final_chunks
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# --------------------------------------------------
# Create embedding model
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# Create Chroma vector store
# --------------------------------------------------

vectorstore = Chroma.from_documents(
    documents=final_chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)


print(f"Stored {len(final_chunks)} chunks in Chroma")