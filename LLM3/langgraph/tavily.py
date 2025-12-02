#pip install tavily
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import TavilySearchAPIRetriever
load_dotenv()
os.environ.get('TAVILY_API_KEY')


retriever = TavilySearchAPIRetriever()
result = retriever.invoke('2025 멜론 뮤직 어워드 시상 내역은 어떻게 되나요')
context = '\n\n--\n\n'.json([doc.page_content for doc in result])

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
('system','너는 음원 분석 전문가 입니다. 다음 수상내역을 참고해서 올해 가장 유망하고 내년에도 유망한 가수를 추천해 주세요.'),
    ('human','질문 : {question} 수상내역 : {context}')
])

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-4o-mini',temperature=0)
chain = prompt | llm | StrOutputParser()
question = '올해 가장 인기 있는 가수는?'
result = chain.invoke({'context':context,'question':question})
print(result)