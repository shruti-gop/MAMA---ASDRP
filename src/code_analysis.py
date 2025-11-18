import pandas as pd

class DataCreation():
    def __init__(self, title:str, tested_values: list):
        self.title= title
        self.research_paper=[1]
        self.question_num=[1,2,3]
        self.tested_value= tested_values
        self.index= pd.MultiIndex.from_product([self.research_paper,self.question_num,self.tested_value],names=["Research_paper","Question",title])
        self.df= pd.DataFrame(columns=["answer_relevancy","faithfulness","answer_correctness"],
        index=self.index)


    def add_data(self, research_paper:int, question:int, tested_value:int, answer_relevancy:float, faithfulness:float, answer_correctness: float):
        new_data= pd.DataFrame({
            "answer_relevancy": [answer_relevancy],
            "faithfulness": [faithfulness],
            "answer_correctness": [answer_correctness]
        },
        index= pd.MultiIndex.from_tuples([(research_paper, question, tested_value)], names=["Research_paper","Question",self.title])
        )
        self.df= pd.concat([self.df, new_data])


