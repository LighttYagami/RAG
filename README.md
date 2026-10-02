Conversational RAG Knowledge Assistant
A conversational Retrieval-Augmented Generation (RAG) assistant built over GitLab's public handbook policies.
The project demonstrates an end-to-end RAG pipeline with structured HTML ingestion, hierarchical chunking, local semantic embeddings, ChromaDB vector search, MultiQuery retrieval, cross-encoder reranking, conversational query routing, source citations, and a Streamlit chat interface.
Features
Ingests public GitLab handbook policy pages from HTML.
Extracts the main article content while removing navigation, scripts, styles, and taxonomy metadata.
Preserves document hierarchy using `h1`, `h2`, and `h3` headings.
Creates structured chunks with section metadata.
Uses `sentence-transformers/all-MiniLM-L6-v2` for local text embeddings.
Stores embeddings and metadata in ChromaDB.
Uses MultiQuery retrieval to improve retrieval recall through alternative query formulations.
Removes duplicate retrieved chunks.
Uses a cross-encoder reranker to improve the relevance of retrieved context.
Routes casual conversation directly to Gemini while sending knowledge-base questions through the RAG pipeline.
Maintains conversational history using LangChain message objects.
Provides source/section information with RAG responses.
Includes a Streamlit web interface.
Architecture
```text
GitLab Public Handbook
        |
        v
     requests
        |
        v
   BeautifulSoup
        |
        v
 Main content extraction
        |
        v
 HTML cleaning
        |
        v
 h1 / h2 / h3 hierarchy extraction
        |
        v
 Structured LangChain Documents
        |
        v
 RecursiveCharacterTextSplitter
        |
        v
 146 final chunks
        |
        v
 Hugging Face Sentence Transformer
(all-MiniLM-L6-v2)
        |
        v
     ChromaDB
        |
        v
     Retriever
        |
        v
    MultiQuery
        |
        v
 Duplicate removal
        |
        v
 Cross-Encoder Reranking
        |
        v
 Top relevant context
        |
        v
 Prompt + Chat History
        |
        v
      Gemini
        |
        v
 Answer + Sources
```
Conversational Routing
The application uses a router before the RAG pipeline.
```text
                    User Question
                         |
                         v
                      Router
                         |
              requires_retrieval?
                  /             \
                No               Yes
                |                 |
                v                 v
          Direct Gemini       RAG Pipeline
                |                 |
                |          MultiQuery Retrieval
                |                 |
                |             Deduplication
                |                 |
                |          Cross-Encoder Reranking
                |                 |
                |          Context Construction
                |                 |
                └────────┬────────┘
                         v
                      Answer
                         |
                         v
                  Update Chat History
```
The router owns the conversation history. The RAG pipeline receives the existing history but does not modify it. After the answer is generated, the router appends the current `HumanMessage` and `AIMessage` to the shared history.
Project Structure
```text
rag_interview_project/
│
├── ingest.py
├── chunker.py
├── vectorstore.py
├── retriever.py
├── rag_pipeline.py
├── router.py
├── llm.py
├── app.py
│
├── chroma_db/
├── .env
├── .gitignore
└── README.md
```
File Responsibilities
`ingest.py`
Downloads the configured GitLab handbook pages and extracts their main content using `requests` and BeautifulSoup.
The ingestion process:
Sends an HTTP request to each configured URL.
Parses the HTML.
Locates the `<main>` element.
Removes scripts, styles, navigation, and taxonomy metadata.
Stores the cleaned HTML content and source URL.
`chunker.py`
Converts the cleaned HTML into structured LangChain `Document` objects.
The chunker:
Detects `h1`, `h2`, and `h3` headings.
Maintains the current heading hierarchy.
Groups paragraphs and list items under their section.
Stores hierarchical metadata such as:
```text
People Policies > General Employment Practices > Open Door Policy
```
Large sections are further split using `RecursiveCharacterTextSplitter`.
Current configuration:
```python
chunk_size=1000
chunk_overlap=150
```
The current corpus produces approximately:
72 structured sections
146 final chunks
`vectorstore.py`
Creates the ChromaDB vector store from the final chunks.
The project uses:
```text
sentence-transformers/all-MiniLM-L6-v2
```
for local embedding generation.
The resulting embeddings and document metadata are persisted in:
```text
./chroma_db
```
`retriever.py`
Loads the existing ChromaDB collection and exposes the retriever.
Current retrieval configuration:
```python
search_type="similarity"
k=8
```
A helper function also removes exact duplicate chunk content before returning the requested number of unique results.
`rag_pipeline.py`
Implements the main RAG generation pipeline.
The pipeline performs:
MultiQuery retrieval.
Candidate collection.
Exact duplicate removal.
Cross-encoder reranking.
Context construction with source and section metadata.
Prompt construction with retrieved context and conversation history.
Gemini generation.
The reranker uses:
```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```
and selects the top 5 candidates after reranking.
`router.py`
Acts as the conversational entry point.
It uses structured output with Pydantic to determine whether the current question requires knowledge-base retrieval.
Example routing:
```text
"Hi"
    -> Direct Gemini

"What happens if an employee violates the acceptable use policy?"
    -> RAG Pipeline

"Does that mean termination?"
    -> RAG Pipeline
```
`router.py` also owns the conversation history:
```python
chat_history = []
```
and appends:
```python
HumanMessage(...)
AIMessage(...)
```
after every completed turn.
`llm.py`
Initializes the Gemini chat model using LangChain's Google Generative AI integration.
The API key is loaded from `.env`.
`app.py`
Provides the Streamlit user interface.
It displays:
User messages
Assistant responses
Retrieved source sections
Source URLs
Conversation history
Retrieval Pipeline
The project uses a two-stage retrieval approach.
Stage 1: Recall-oriented retrieval
MultiQueryRetriever generates alternative formulations of the user's question and retrieves candidate documents using vector similarity.
This helps when the wording of the user's question differs from the wording used in the source documents.
Stage 2: Precision-oriented reranking
The retrieved candidates are passed to a cross-encoder:
```text
Query + Candidate Chunk
        |
        v
Cross Encoder
        |
        v
Relevance Score
```
The candidates are ranked against the original user question, and the top 5 are passed to the generation step.
This separates:
```text
Embedding retrieval → high recall
Cross-encoder reranking → higher precision
```
Conversation History
Conversation history is maintained separately from the knowledge base.
The project uses LangChain message objects:
```python
HumanMessage(...)
AIMessage(...)
```
The history is inserted into prompts using:
```python
MessagesPlaceholder(variable_name="chat_history")
```
The basic flow is:
```text
User Question
      |
      v
Router
      |
      v
RAG / Direct Gemini
      |
      v
Answer
      |
      v
Append HumanMessage + AIMessage
      |
      v
Updated chat_history
```
The ChromaDB vector store contains the knowledge base. It does not store the conversation history.
Source Grounding
Each chunk retains metadata including:
```python
{
    "source": "...",
    "section_path": "...",
    "h1": "...",
    "h2": "...",
    "h3": "..."
}
```
The retrieved metadata is included in the generation context and displayed in the UI so that responses can be traced back to the relevant handbook section.
Tech Stack
Python
LangChain
ChromaDB
Gemini
Hugging Face / Sentence Transformers
BeautifulSoup
Streamlit
Pydantic
Setup
1. Clone the repository
```bash
git clone <your-repository-url>
cd rag_interview_project
```
2. Create and activate a virtual environment
Windows PowerShell:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
3. Install dependencies
Install the packages used by the project, for example:
```bash
pip install langchain
pip install langchain-community
pip install langchain-text-splitters
pip install langchain-chroma
pip install langchain-huggingface
pip install langchain-google-genai
pip install sentence-transformers
pip install beautifulsoup4
pip install requests
pip install lxml
pip install streamlit
pip install pydantic
pip install langchain-classic
```
4. Configure the Gemini API key
Create a `.env` file:
```env
GOOGLE_API_KEY=your_actual_api_key_here
```
Do not commit `.env` to Git.
5. Build the knowledge base
Run the ingestion/chunking/vector-store pipeline in the project order.
The vector database will be created locally under:
```text
chroma_db/
```
6. Run the Streamlit application
```bash
python -m streamlit run app.py
```
Example Questions
Knowledge-base questions:
```text
What is the purpose of the acceptable use policy?

What kinds of activities are prohibited under the acceptable use policy?

What happens if an employee violates the acceptable use policy?

Does the policy mention disciplinary action?
```
Conversational questions:
```text
Hi

Hello

Thanks

How are you?
```
Follow-up questions can use previous conversation context:
```text
What happens if an employee violates the acceptable use policy?

Does that mean termination?
```
Design Decisions
Why HTML instead of PDF?
The source material is available as structured public HTML. HTML preserves heading hierarchy and page structure, making it easier to create meaningful sections and metadata.
Why hierarchical chunking?
Instead of blindly splitting the entire webpage by character count, the project first creates sections based on the document's heading hierarchy.
This allows metadata such as:
```text
H1 > H2 > H3
```
to remain attached to the retrieved chunk.
Large sections are then split using recursive character splitting.
Why embeddings?
Keyword matching depends heavily on exact word overlap. Embeddings provide a semantic representation that allows the retrieval system to find text that is conceptually related even when the wording differs.
Why ChromaDB?
Chroma provides a vector-store abstraction for storing embeddings together with document metadata and performing similarity search.
Why MultiQuery?
A single user query may not match the wording of the relevant document closely enough. MultiQuery generates alternative formulations to improve retrieval recall.
Why a cross-encoder reranker?
Embedding similarity is useful for retrieving candidates efficiently, but a cross-encoder can examine the query and candidate text together to make a more focused relevance judgment.
Why separate chat history from the vector database?
The knowledge base and conversation history serve different purposes:
```text
Knowledge base
→ persistent source information

Chat history
→ current conversational context
```
Keeping them separate also prevents conversational messages from becoming part of the searchable policy corpus.
Current Scope
This project intentionally focuses on a compact conversational RAG system.
It does not currently implement:
LangGraph
MCP
Agentic tool calling
Contextual compression
FastAPI backend
Production database-backed conversation persistence
RAG evaluation framework
Distributed deployment
These can be added as future extensions if required.
Future Improvements
Potential improvements include:
Add contextual compression after reranking.
Add relevance-score thresholds and abstention when retrieved context is insufficient.
Add automated RAG evaluation for retrieval recall and answer faithfulness.
Persist conversation history using an external store.
Expose the RAG pipeline through a FastAPI service.
Convert the workflow into a LangGraph stateful workflow.
Add MCP-based tool interfaces.
Add more policy pages to the knowledge base.