from typing import TypedDict
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
import pypdf
from ragas import evaluate
from ragas.metrics import (faithfulness,answer_relevancy,context_precision,context_recall,answer_similarity,answer_correctness)
import pandas as pd
from datasets import Dataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()
Chunk_size=1000
Chunk_overlap=200
k_value=4
# We can decide the specifc path later
openai_api_key = os.getenv("OPENAI_API_KEY")
class MessageDict(TypedDict):
    user_question: str
    paper_path: str
    vector_store: object
    routing_strategy: str
    context: str
    sent_context: str
    agent_response: str
    final_response: str
    

class MultiAgentSystem():
    def __init__(self, openai_api_key: str, chunk_size: int = Chunk_size, chunk_overlap: int = Chunk_overlap, int=k_value):
        self.llm_orchestrator= ChatOpenAI(model="gpt-4.1-nano", temperature=0, openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0, openai_api_key=openai_api_key)
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_value= k_value

    def create_vector_store(self, paper_path: str) -> FAISS:
        loader = PyPDFLoader(paper_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,)
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
        docs = vector_store.similarity_search(user_question, k=k_value)
        context = "\n".join([doc.page_content for doc in docs])
        message_dict['context'] = context
        prompt = f"""

         Analyze the query: {user_question} 
        Provided with the context: {context}.
         Classify it into ONE of the following categories:

         Agent2: Use when the query asks about the following:
            - Evidence, data, or experiments used in the research paper
            - Methodology, experimental setup, or a procedure that was used in the research paper
            - Statistical analysis or quantitative results presented in the research paper
            - Dataset details or evaluation approaches.
        Agent3: Use when the query asks about the following:
            - The main claim or hypothesis of the research paper or the conclusions drawn by the authors
            - General nature or purpose of the research paper along with its contributions and background.
            - Implications, significance, or potential applications of the research findings.
            - Limitations or future work that the author(s) suggest in the research paper.
            - If the query asks about reasoning, justification, or analysis that supports a claim which is qualitative and not specific to data or methods, as that falls under Agent2.
        CRITICAL RULES:
- If the query mentions specific metrics, scores, or numerical comparisons → Agent2
- If the query uses words like "compare" with data/results → Agent2
- If the query asks "how" something works conceptually (no metrics) → Agent3
- If the query asks "why" about research motivation/implications → Agent3
        IMPORTANT:
        Your response must be only one word: either 'Agent2' or 'Agent3'.
         If the query does not fit into either of these categories, respond with 'Unanswerable'."""
        response = self.llm_orchestrator.invoke(prompt)
        message_dict['routing_strategy']=response
        agent_response = response.content.strip()
        message_dict['agent_response'] = agent_response
        
        context_prompt= f"""

        You are an expert research analyst and critical thinker who is tasked with extracting and summarizing the most crucial and relevant information from research papers.

        Provided with the research paper context: {context}, 
        Provided the agent decision: {agent_response}, 
        and the user question: {user_question},
        generate a descriptive context that can be passed down to the subsequent 
        agent to ensure that they have the required information to answer the user's question 
        effectively and accurately. 

        Information Of Structure:
        - All of the information provided in the context must be from the provided context and not fabricated.
        - The generated context must be concise yet comprehensive, capturing all essential details that are pertinent to the user question, with the location of such details in the research paper to refrence For example: figures, tables, exact data curations.
        - Ensure the usage of clear, precise and exact details from the passed in context.

        if {agent_response} is 'Agent2', 
        ensure that the context focuses on:
            - Evidence, data, experiments, methodology, procedures, statistical analysis, and quantitative details presented in the research paper that are important and directly relevant to the user question.
            - Methodology, experimental setup, or a procedure that was used in the research paper.
            - If applicable, include dataset details, evaluation approaches, and specific numerical details.

        if {agent_response} is 'Agent3', ensure that the context focuses on while mantaining depth and relevance to the user question:
            - The main claim or hypothesis of the research paper or the conclusions drawn by the authors
            - General nature or purpose of the research paper
            - Implications, significance, or potential applications of the research findings
            - Limitations or future work that the author(s) suggest in the research paper
            - If the query asks about reasoning, justification, or analysis that supports a claim, which is qualitative and not specific to data or methods.

        IMPORTANT: The context generated must be highly relevant to the user's question and in-depth, including all relevant and crucial parts that can help agents 2 or 3 effectively answer the user's question using the context that you will generate.
        Output Format:
        - Provide the generated context in clear, coherent in 400-600 words.
        - Ensure that the structure of the generated context is a bullet point format with exact refrences and if applicable numerical data from the research paper to support the query.
        """
        message_dict["sent_context"] = self.llm.invoke(context_prompt).content.strip()
        return agent_response

    #"Agent 2 is called by Agent 1 if the user question is related to understanding the evidence within the research paper passed in."
    def agent_2(self, message_dict: MessageDict) -> str:
        sent_context= message_dict['sent_context']
        user_question= message_dict['user_question']
        prompt= f"""
        You are an expert research analyst and critical thinker who is tasked with extracting and explaining evidence from research papers.
        Given the following question:{user_question}.
        Research Paper Context:{sent_context}.
        
        You are tasked with extracting and explaining the relevant details based off the usser question from the research paper context:
        1. Identify and extract the most relevant evidence, data, or experiments from the research paper that directly address the user question.
        2.Make sure to provide specific details by providing exact references and numerical data from the research paper to support your answer.
        3. Explain why the extracted evidence is appropriate and how it relates to the user question.
        4. If the query is regarding:
            - Methodology, experimental setup, or a procedure that was used in the research paper ensures the provided evidence focuses on those aspects and references the specific section and details from the research paper that identify the protocols used.
            - If applicable, include dataset details, evaluation approaches, and specific numerical details. For example: figures, tables, and exact data curations.
        5. If you have no relevant evidence, output 'No evidence found for the passed in query.'

        Format your response as follows:
        - Write a concise 250-500 word summary of the relevant evidence with exact references and numerical data from the research paper(If applicable) that directly addresses the user question.
        - Provide a brief explanation of why this evidence is appropriate and how it relates to the user question.
        - Ensure clarity and flow in your response.
        """    
        response = self.llm.invoke(prompt)
        agent_response = response.content.strip() 
        message_dict['agent_response'] = agent_response  
        return agent_response
    
    #"Agent 3 is called by Agent 1 if the user question is related to analyzing and understanding the nature of the research paper passed in, along with its "
    def agent_3(self, message_dict: MessageDict) -> str:
        sent_context= message_dict['sent_context']
        user_question= message_dict['user_question']
        prompt= f"""
        You are an expert research analyst and critical thinker.
        Task:
        You are tasked with extracting and explaining the relevant details based off the usser question from the research paper context:
            Based off of the query make sure to integrate the following, as appropriate:
            1. The main claim or hypothesis of the research paper or the conclusions drawn by the authors
            2. General nature or purpose of the research paper
            3. Implications, significance, or potential applications of the research findings
            4. Limitations or future work that the author(s) suggest in the research paper
            5. justification, or analysis that supports a claim, which is qualitative and not specific to data or methods.

        Answer the following question:{user_question}.
        Research Paper Context:{sent_context}.
    
        Format your response as follows:
        - Write a concise 250-500 word summary of the relevant extracted details with exact references and qualitiative information from the research paper(If applicable) that directly addresses the user question.
        - Provide a brief explanation of why this evidence is appropriate and how it relates to the user question.
        - Ensure clarity and flow in your response.
"""
        response = self.llm.invoke(prompt)
        agent_response = response.content.strip() 
        message_dict['agent_response'] = agent_response
        return agent_response
    
    # This agent essentially just finalizes the response and evaluates it to ensure it apporpriateness relative to the user question.
    def agent_4(self, message_dict: MessageDict) -> str:
        agent_response= message_dict['agent_response']
        user_question= message_dict['user_question']
        paper_path = message_dict['paper_path']
        context = message_dict['context']
        prompt= f"""
        Based on the query '{user_question}' use the {context} to evaluate the following 
        response outputted by the previous agent: {agent_response}.
        
        Important:
           - Ensure to use proper reasoning to evaluate the response based on whether it includes the right information and follows the metric as appropriate.
           - If the {message_dict['routing_strategy']} was agent2 check to see if the evidence provided is relevant 
            and accurate. If it was agent3, ensure that the analysis of the text is thorough and well-supported by the 
            context and properly explains the reasonings."

        Use the following Metric to do so:
        if {message_dict['routing_strategy']} is 'Agent2', 
         1. **Relevance (0-0.3):** Does the response identify relevant evidence, data, experiments, or methodology?
         2. **Specificity (0-0.3):** Does it include exact numbers, metrics, dataset details, or specific references?
         3. **Accuracy (0-0.2):** Is the information factually correct based on the context?
         4. **Completeness (0-0.2):** Does it fully address all aspects of the questi
        if {message_dict['routing_strategy']} is 'Agent3', ensure that the context focuses on while mantaining depth and relevance to the user question:
         1. **Relevance (0-0.3):** Does the response address the main claims, implications, or reasoning asked about?
         2. **Depth (0-0.3):** Does it provide thorough analysis of the authors' reasoning and justification?
         3. **Accuracy (0-0.2):** Is the interpretation faithful to what the authors actually state?
         4. **Completeness (0-0.2):** Does it cover claims, implications, limitations, or future work as relevant?

        Format:
          Provide a score that assesses how well the previous agent's response answers the user question on a scale of 0 to 1, 
          with 1 identifying the rsponse as that of following the metric perfectly. .7 meaning that the response encompassed most of the parts of the metric.
          .5 meaning that the response is poor and not up to the standards of the metric.
          """
        response = self.llm.invoke(prompt)
        agent_response = response.content.strip()   
        return agent_response
    

    def run(self, paper_path: str,user_question: str='') -> MessageDict:
        message_dict: MessageDict = {
            'paper_path': paper_path,
            'vector_store': self.create_vector_store(paper_path),
            'user_question': user_question,
            'context':'',
            "sent_context": '',
            'routing_strategy':'',
            'agent_response': '',
            'final_response': '',
        }   

        agent_1_response = self.agent_1(message_dict)

        if agent_1_response.lower() == 'agent2':
            message_dict['routing_strategy']='Agent2'
            agent_2_response = self.agent_2(message_dict)
            message_dict['agent_response'] = agent_2_response

        elif agent_1_response.lower() == 'agent3':
            message_dict['routing_strategy']='Agent3'
            agent_3_response = self.agent_3(message_dict)
            message_dict['agent_response'] = agent_3_response

        else:
            print(agent_1_response)  
        
        message_dict['final_response'] = self.agent_4(message_dict)
        return message_dict
    
    def dataset_for_evaluation(self, user_question:str, agent_response:str, context:str):
        context_list = [chunk.strip() for chunk in context.split('\n') if chunk.strip()]
        dataset={
                "question": [user_question],
                "answer": [agent_response],
                "contexts": [context_list]
            }
        
        ragas_llm = LangchainLLMWrapper(self.llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(self.embeddings)
        
        evaluation_dataset = Dataset.from_dict(dataset)
        result= evaluate(
            evaluation_dataset,
            metrics=[
                faithfulness,
                answer_relevancy, 
            ], 
                llm=ragas_llm,
                embeddings=ragas_embeddings              
              
        )
        return result.to_pandas()
    

    




# Using the Code To FINALLY get results!!:

multi_agent_system = MultiAgentSystem(openai_api_key=openai_api_key)
paper_path = r"C:\Users\geeta\OneDrive\Desktop\Research_Paper\researchpaper_1.pdf"  
message_dict = multi_agent_system.run(paper_path, user_question= "How do the F1 scores compare between the DeClarE configuration and the other configurations that require manual intervention?")
results= multi_agent_system.dataset_for_evaluation(message_dict['user_question'],message_dict['agent_response'],message_dict['context'])

print("User Question:", message_dict['user_question'])
print("routing strategy:", message_dict['routing_strategy'])
print("agent response:", message_dict['agent_response'])
print("Final Response:", message_dict['final_response'])
print(results)

































# print("User Question:", message_dict['user_question'])
# print("routing strategy:", message_dict['routing_strategy'])
# print("agent response:", message_dict['agent_response'])
# print("Final Response:", message_dict['final_response'])
# print("Evaluation Results:\n", results)