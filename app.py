import streamlit as st

from router import route_query


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="GitLab Policy Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📚 GitLab Policy Assistant")

st.caption(
    "A conversational RAG assistant over GitLab's public handbook policies."
)


# --------------------------------------------------
# Chat history for UI
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Display previous messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("Sources"):

                for source in message["sources"]:
                    st.markdown(
                        f"- **{source['section']}**  \n"
                        f"{source['url']}"
                    )


# --------------------------------------------------
# User input
# --------------------------------------------------

question = st.chat_input(
    "Ask something about GitLab's policies..."
)


if question:

    # Display user message
    st.chat_message("user").markdown(question)

    # Get answer from router
    answer, documents = route_query(question)

    # Display assistant response
    with st.chat_message("assistant"):

        st.markdown(answer)

        if documents:

            with st.expander("Sources"):

                for doc in documents:

                    st.markdown(
                        f"- **{doc.metadata.get('section_path')}**  \n"
                        f"{doc.metadata.get('source')}"
                    )


    # Save conversation to Streamlit UI
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": [
                {
                    "section": doc.metadata.get("section_path"),
                    "url": doc.metadata.get("source")
                }
                for doc in documents
            ]
        }
    )