from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from retriever import retriever
from llm import llm


load_dotenv()


# --------------------------------------------------
# 1. Multi-query retriever
# --------------------------------------------------

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)


# --------------------------------------------------
# 2. Reranker
# --------------------------------------------------

reranker_model = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

reranker = CrossEncoderReranker(
    model=reranker_model,
    top_n=5
)


# --------------------------------------------------
# 3. RAG prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant answering questions using a
retrieval-augmented knowledge base.

Use the retrieved context to answer the user's question.

If the retrieved context does not contain enough information,
say that you do not have enough information.

Do not invent facts.

Retrieved context:
{context}
"""
    ),

    MessagesPlaceholder(variable_name="chat_history"),

    (
        "human",
        "{question}"
    )
])


# --------------------------------------------------
# 4. Build context
# --------------------------------------------------

def build_context(documents):

    context_parts = []

    for i, doc in enumerate(documents):

        context_parts.append(
            f"""SOURCE {i + 1}
Section: {doc.metadata.get("section_path")}
Source: {doc.metadata.get("source")}

Content:
{doc.page_content}
"""
        )

    return "\n\n".join(context_parts)


# --------------------------------------------------
# 5. RAG pipeline
# --------------------------------------------------

def rag_pipeline(question, chat_history):

    # Step 1:
    # Generate multiple queries and retrieve candidates
    documents = multi_query_retriever.invoke(question)

    # Step 2:
    # Remove exact duplicate chunks
    unique_documents = []
    seen_chunks = set()

    for doc in documents:

        chunk_text = doc.page_content.strip()

        if chunk_text in seen_chunks:
            continue

        seen_chunks.add(chunk_text)
        unique_documents.append(doc)

    # Step 3:
    # Rerank candidates using the original question
    reranked_documents = reranker.compress_documents(
        documents=unique_documents,
        query=question
    )

    # Step 4:
    # Build context
    context = build_context(reranked_documents)

    # Step 5:
    # Create RAG prompt
    messages = prompt.format_messages(
        context=context,
        chat_history=chat_history,
        question=question
    )

    # Step 6:
    # Generate answer
    response = llm.invoke(messages)

    answer = response.text

    return answer, reranked_documents