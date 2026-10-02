from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)


retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8}
)


def retrieve_documents(query, max_results=5):

    results = retriever.invoke(query)

    unique_results = []
    seen_chunks = set()

    for doc in results:

        chunk_text = doc.page_content.strip()

        if chunk_text in seen_chunks:
            continue

        seen_chunks.add(chunk_text)
        unique_results.append(doc)

        if len(unique_results) == max_results:
            break

    return unique_results