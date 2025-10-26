from typing import TypedDict, Annotated, Literal
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import operator
import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
class MessageDict(TypedDict):
    user_question: str
    paper_path: str
    vector_store: object
    routing_strategy: str
    context: str
    agent_response: str
    evaluation: dict
    final_response: str
    messages: list[HumanMessage | AIMessage]

class MultiAgentSystem():
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=openai_api_key)
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    def create_vector_store(self, paper_path: str) -> FAISS:
        loader = PyPDFLoader(paper_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(docs, self.embeddings)
        return vector_store




