from langsmith import Client
import os
from dotenv import load_dotenv
load_dotenv()

# 연결 텍스트
client = Client()
print("LangSmith 연결 성공")
print(f"현재 프로젝트 : {os.getenv('LANGCHAIN_PROJECT','default')}")

# 자동 추적
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# llm 생성
llm = ChatOpenAI(model='gpt-4o-mini')
# 프롬프트 입력
prompt = ChatPromptTemplate.from_template('''정렬:{question}''')
# 체인 구성
from langchain_core.output_parsers import StrOutputParser
chain = prompt | llm | StrOutputParser()
# 결과 받기
result = chain.invoke({'question':'RAG란?'})
# 결과 출력
print(result)

# 커스텀 추적
from langsmith.run_helpers import traceable
@traceable(name='custom_rag_pipeline')
def my_rag_function(question:str) -> str:
    result = chain.invoke({'question' : question})

reault = my_rag_function('langchain이란?')
print(result)