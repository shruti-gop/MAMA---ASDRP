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
from ragas.metrics import (faithfulness,answer_relevancy,answer_correctness)
import pandas as pd
from datasets import Dataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import code_analysis


k_vals=[4,5,6,7,8]
k_value_df= code_analysis.DataCreation("k value",k_vals)
chunking_overs=[90,120,150,180,210]
chunking_overs_df=code_analysis.DataCreation("Chunking Overlap",chunking_overs)
chunking_sizes= [900,1200,1500,1800,2100]
chunking_sizes_df=code_analysis.DataCreation("Chunking Size", chunking_sizes)
chunking_models=["gpt-4.1-nano","gpt-4.1-mini","gpt-4.1"]
chunking_models_df=code_analysis.DataCreation("Models",chunking_models)

questions=[
    "How do the F1 scores compare between the DeClarE configuration and the other configurations that require manual intervention?",
    "The authors claim that “our method does not require any feature-engineering, lexicons, or other manual intervention” (Popat et al.). What is the significance of this in regards to the limitations of prior methods, and what are the differences in design that make this possible?",
    "What was the setup under which the experiment was performed for evaluating their approach and providing evidence for their approach?"  
]
ground_truths=[
    """
The results in terms of F1 score were Distant Supervision configuration resulting in the highest for the Snopes dataset with 0.82 and DeClarE (full) with 0.79. 
Although DeClarE scores lower, it does have the advantage of not requiring manual intervention which makes it more applicable for larger use cases. 
In the PolitiFact dataset, the DeClarE scored the highest F1 score with 0.68, while the highest manual intervention configuration scored 0.64 (CNN-text) in that same dataset.
""",
"""
The limitations of the prior methods were the required manual feature engineering and lexicons. 
As the style of the text is altered, these manual methods do not work and therefore are not able to be generalized. 
DeClarE allows the method to not use any manual intervention like conventional methods through looking at the correlation between the claim and its supporting articles through an attention mechanism. With this method it can determine the accuracy of misinformation without the help of manual intervention such as lexicons, making the model more adaptable across different contexts.
""",
"""
Used the Snopes, PolitiFact and NewsTrust datasets. Reserved 10% of the data as validation data for parameter tuning. 
Reported 10-fold cross validation results on the remaining 90% of the data. 
The model is trained on 9-folds and the remaining fold is used as test data.
"""
]

load_dotenv()
Chunk_size=1500
Chunk_overlap=150
k_value=6

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
    def __init__(self, openai_api_key: str, model: str, chunk_size: int= Chunk_size, chunk_overlap: int=Chunk_overlap, k_value: int=k_value):        
        self.llm_orchestrator= ChatOpenAI(model=model, temperature=0, openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(model=model, temperature=0, openai_api_key=openai_api_key)
        self.llm_ground=ChatOpenAI(model=model, temperature=0, openai_api_key=openai_api_key)
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
        docs = vector_store.similarity_search(user_question, k=self.k_value)
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
        RULES:
        - If the query mentions specific metrics, scores, or numerical comparisons ==> Agent2
        - If the query uses words like "compare" with data/results ==> Agent2
        - If the query asks "how" something works conceptually (no metrics) ==> Agent3
        - If the query asks "why" about research motivation/implications ==> Agent3
        IMPORTANT:
        Your response must be only one word: either 'Agent2' or 'Agent3'.
         If the query does not fit into either of these categories, respond with 'Unanswerable'."""
        response = self.llm_orchestrator.invoke(prompt)
        message_dict['routing_strategy']=response
        agent_response = response.content.strip()
        message_dict['agent_response'] = agent_response
        
        return agent_response

    #"Agent 2 is called by Agent 1 if the user question is related to understanding the evidence within the research paper passed in."
    def agent_2(self, message_dict: MessageDict) -> str:
        sent_context= message_dict['context']
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
        - Write a concise 150-200 word summary of the relevant evidence with exact references and numerical data from the research paper(If applicable) that directly addresses the user question.
        - Provide a brief explanation of why this evidence is appropriate and how it relates to the user question.
        - Ensure clarity and flow in your response.
        """    
        response = self.llm.invoke(prompt)
        agent_response = response.content.strip() 
        message_dict['agent_response'] = agent_response  
        return agent_response
    
    #"Agent 3 is called by Agent 1 if the user question is related to analyzing and understanding the nature of the research paper passed in, along with its "
    def agent_3(self, message_dict: MessageDict) -> str:
        sent_context= message_dict['context']
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
        - Write a concise 150-200 word summary of the relevant extracted details with exact references and qualitiative information from the research paper(If applicable) that directly addresses the user question.
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
            You are an expert research analyst. You have access to the entire research paepr.
            Research Paper: {context}

            Question: {user_question}
             Task: Provide a comprehensive, accurate answer to this question based off of the whole research paper being provided.
             - Include specifc and exact details: Like as much quantative data as possible. 
             - Be thorough and percise
             - Make sure to cite specific sections/page numbers whenever appropriate and relevant
             - Length: 150-250 words.
        Important:
        This response will serve as a refrence ground truth for evaluation purpose.
          """
        response = self.llm_ground.invoke(prompt)
        message_dict["final_response"]= response.content.strip()   
        return response.content.strip()
    

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
        
        #message_dict['final_response'] = self.agent_4(message_dict)
        return message_dict
    
    def dataset_for_evaluation(self, user_question:str, agent_response:str, context:str, ground_truth:str):
        context_list = [chunk.strip() for chunk in context.split('\n') if chunk.strip()]
        dataset={
                "question": [user_question],
                "answer": [agent_response],
                "contexts": [context_list],
                "reference":[ground_truth]
            }
        
        ragas_llm = LangchainLLMWrapper(self.llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(self.embeddings)
        
        evaluation_dataset = Dataset.from_dict(dataset)
        result= evaluate(
            evaluation_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                answer_correctness 
            ], 
                llm=ragas_llm,
                embeddings=ragas_embeddings              
              
        )
        return result.to_pandas()
    

    




# Using the Code To FINALLY get results!!:


paper_path = r"C:\Users\geeta\OneDrive\Desktop\Research_Paper\researchpaper_1.pdf"  


for quer in range(3):
    question=questions[quer]
    for m in chunking_models:
        multi_agent_system = MultiAgentSystem(openai_api_key=openai_api_key,model=m)
        message_dict=multi_agent_system.run(paper_path,user_question=question)
        results=multi_agent_system.dataset_for_evaluation(user_question=question,agent_response=message_dict['agent_response'], context=message_dict['context'],ground_truth=ground_truths[quer])
        faithfulness=results["faithfulness"].iloc[0]
        answer_relevancy=results["answer_relevancy"].iloc[0]
        answer_correctness=results["answer_correctness"].iloc[0]
        chunking_models_df.add_data(1,quer,m,answer_relevancy,faithfulness,answer_correctness)
    print(chunking_models_df.df)
  

    

