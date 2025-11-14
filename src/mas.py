from typing import TypedDict
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
import pypdf
load_dotenv()
Chunk_size=1000
Chunk_overlap=200
# We can decide the specifc path later
openai_api_key = os.getenv("OPENAI_API_KEY")
class MessageDict(TypedDict):
    user_question: str
    paper_path: str
    vector_store: object
    routing_strategy: str
    context: str
    agent_response: str
    final_response: str
    

class MultiAgentSystem():
    def __init__(self, openai_api_key: str, chunk_size: int = Chunk_size, chunk_overlap: int = Chunk_overlap):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def create_vector_store(self, paper_path: str) -> FAISS:
        loader = PyPDFLoader(paper_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        docs = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(docs, self.embeddings)
        return vector_store
    
  #  """Essentially, Agent 1 retrieves the releveant conetent from the 
   #     passed in document appropriately chunks it and then
    #    uses that to answer the user question. It then decided if 
     #   Agent 2 or Agent 3 need to be called based on the queston being asked by the use.r"""
    

    def agent_1(self, message_dict: MessageDict) -> str:
        vector_store = message_dict['vector_store']
        user_question = message_dict['user_question']
        message_dict['user_question'] = user_question
        docs = vector_store.similarity_search(user_question, k=4)
        context = "\n".join([doc.page_content for doc in docs])
        message_dict['context'] = context
        prompt = f"Using the following context:{context}, anayze the query: {user_question} and decide the appropriate Agent to use. If the passed in query is related to understanding the evidence used within the research paper, respond with 'Agent2'. If the the query is asking about the claim and general nature of the research paper, then respond with 'Agent3'. If the query cannot be answered using the two agents, then respond with 'please pass in appropriate query'. Your response must be only one word"
        response = self.llm.invoke(prompt)
        agent_response = response.content.strip()
        message_dict['agent_response'] = agent_response
        return agent_response

    #"Agent 2 is called by Agent 1 if the user question is related to understanding the evidence within the research paper passed in."
    def agent_2(self, message_dict: MessageDict) -> str:
        context= message_dict['context']
        user_question= message_dict['user_question']
        prompt= f"Based on the query '{user_question}' in detail provide the relevant evidence from the context: {context} and provice why it is appropriate to the passed in query. If you have no relevant evidence, output 'No evidence found for the passed in query.'"    
        response = self.llm.invoke(prompt)
        agent_response = response.content.strip()   
        return agent_response
    #"Agent 3 is called by Agent 1 if the user question is related to analyzing and understanding the nature of the research paper passed in, along with its "
    def agent_3(self, message_dict: MessageDict) -> str:
        context= message_dict['context']
        user_question= message_dict['user_question']
        prompt= f"Based on the query '{user_question}' with thorough detail analyze the claim made within the research paper using the context: {context} and provide a comprehensive response to the user question."
        response = self.llm.invoke(prompt)
        agent_response = response.content.strip() 
        message_dict['agent_response'] = agent_response
        return agent_response
    # This agent essentially just finalizes the response and evaluates it to ensure it apporpriateness relative to the user question.
    def agent_4(self, message_dict: MessageDict) -> str:
        agent_response= message_dict['agent_response']
        user_question= message_dict['user_question']
        paper_path = message_dict['paper_path']
        loader = PyPDFLoader(paper_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        docs = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(docs, self.embeddings)
        docs = vector_store.similarity_search(user_question, k=4)
        context = "\n".join([doc.page_content for doc in docs])
        prompt= f"Based on the query '{user_question}' use the {context} to evaluate the following response outputted by the previous agent: {agent_response}. Then provide a score that assesses how well the previous agent's response answers the user question on a scale of 1 to 10, with 10 being the highest. Provide reasoning for giving the score and suggest any improvements if necessary."
        response = self.llm.invoke(prompt)
        agent_response = response.content.strip()   
        return agent_response
    

    def run(self, paper_path: str,user_question: str='') -> MessageDict:
        message_dict: MessageDict = {
            'paper_path': paper_path,
            'vector_store': self.create_vector_store(paper_path),
            'user_question': user_question,
            'routing_strategy': '',
            'context': '',
            'agent_response': '',
            'final_response': '',
            'messages': [],
            "latency": 0.0,
            "tokens_used": 0,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap 
        }   

        agent_1_response = self.agent_1(message_dict)
        if agent_1_response.lower() == 'agent2':
            agent_2_response = self.agent_2(message_dict)
            message_dict['agent_response'] = agent_2_response
        elif agent_1_response.lower() == 'agent3':
            agent_3_response = self.agent_3(message_dict)
            message_dict['agent_response'] = agent_3_response
        else:
            print(agent_1_response)  
        final_response = self.agent_4(message_dict)
        message_dict['final_response'] = final_response
        return message_dict



multi_agent_system = MultiAgentSystem(openai_api_key=openai_api_key)
paper_path = r"C:\Users\geeta\OneDrive\Desktop\Research_Paper\researchpaper_1.pdf"  
message_dict = multi_agent_system.run(paper_path,"What evidence does the paper provide to support its main claim?")
print("Final Response:", message_dict['final_response'])