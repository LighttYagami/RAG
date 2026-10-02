from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

# response = llm.invoke(
#     "Explain what retrieval augmented generation is in one sentence."
# )

# print(response.content)