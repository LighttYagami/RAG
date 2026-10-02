from dotenv import load_dotenv

from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from llm import llm
from rag_pipeline import rag_pipeline


# --------------------------------------------------
# 1. Query routing schema
# --------------------------------------------------

class QueryRoute(BaseModel):

    requires_retrieval: bool = Field(
        description=(
            "Whether the user's current message requires "
            "information from the knowledge base."
        )
    )


# --------------------------------------------------
# 2. Router prompt
# --------------------------------------------------

router_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a query router for a conversational RAG system.

Determine whether the user's CURRENT message requires information
from the knowledge base.

Set requires_retrieval to FALSE for:
- greetings
- salutations
- thanks
- simple casual conversation
- messages that can be answered without the knowledge base

Set requires_retrieval to TRUE when:
- the user asks about information contained in the knowledge base
- the user asks a knowledge-base question
- the user asks a follow-up question that depends on information
  discussed in the conversation

A greeting combined with a knowledge-base question requires retrieval.

Examples:

"Hi"
-> false

"Thanks"
-> false

"Hello, how are you?"
-> false

"What happens if an employee violates the acceptable use policy?"
-> true

"Hey, what does the acceptable use policy say?"
-> true

"Does that mean termination?"
-> true
"""
    ),

    MessagesPlaceholder(variable_name="chat_history"),

    (
        "human",
        "{question}"
    )
])


# --------------------------------------------------
# 3. Structured router
# --------------------------------------------------

structured_llm = llm.with_structured_output(QueryRoute)

router_chain = router_prompt | structured_llm


# --------------------------------------------------
# 4. Direct conversation prompt
# --------------------------------------------------

direct_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful conversational assistant.

Answer casual conversation naturally and briefly.

Do not invent information from the knowledge base.
"""
    ),

    MessagesPlaceholder(variable_name="chat_history"),

    (
        "human",
        "{question}"
    )
])


# --------------------------------------------------
# 5. Chat history
# --------------------------------------------------

chat_history = []


# --------------------------------------------------
# 6. Query router
# --------------------------------------------------

def route_query(question):

    # Determine whether retrieval is required
    route = router_chain.invoke(
        {
            "question": question,
            "chat_history": chat_history
        }
    )

    if route.requires_retrieval:

        answer, documents = rag_pipeline(
            question,
            chat_history
        )

    else:

        messages = direct_prompt.format_messages(
            question=question,
            chat_history=chat_history
        )

        response = llm.invoke(messages)

        answer = response.text

        documents = []


    # --------------------------------------------------
    # Update conversation history ONCE
    # --------------------------------------------------

    chat_history.append(
        HumanMessage(content=question)
    )

    chat_history.append(
        AIMessage(content=answer)
    )

    return answer, documents


# --------------------------------------------------
# 7. Terminal chat
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 80)
    print("RAG CHAT")
    print("=" * 80)

    print("Type 'exit' to quit.\n")

    while True:

        question = input("You: ")

        if question.lower() == "exit":
            break

        answer, documents = route_query(question)

        print("\nAI:", answer)

        if documents:

            print("\nSources:")

            for i, doc in enumerate(documents):

                print(
                    f"{i + 1}. "
                    f"{doc.metadata.get('section_path')} "
                    f"→ {doc.metadata.get('source')}"
                )

        print("\n" + "-" * 80)