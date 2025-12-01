# 필요라이브러리 설치
import os
import warnings
warnings.filterwarnings("ignore")

from typing import List, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv

# LangChain 관련 임포트
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# LangGraph 관련 임포트
from langgraph.graph import StateGraph, START, END

# 환경설정
load_dotenv()

if not os.environ.get('OPENAI_API_KEY'):
    raise ValueError('key check')

# 조건부 엣지가 포함된 그래프
def conditional_graph():
    '''
    조건부 엣지가 포함된 LangGraph
    검색 결과에 따라 다른 경로로 분기
    '''
    # 상태를 정의
    class ConditionalState(TypedDict):
        question : str
        documents : List[Document]
        search_type : str
        answer : str
    
    # 내부문서 LoadText or ...
    INTERNAL_DOCS = {
        "회사" : [Document(page_content="우리 회사의 AI 전략은 RAG 시스템 구축입니다.")],
        "정책" : [Document(page_content="사내 데이터 보안 정책은 외부 공유 금지입니다.")],
    }
    
    # 노드함수들을 구현
    def internal_search_node(state:ConditionalState) -> dict:
        '''
        내부 문서 검색 -> 여기서 Retriever 작동시킴
        '''
        question = state['question']
        documents = []
        for keyword,docs in INTERNAL_DOCS:
            if keyword in question:
                documents.extend(docs)
        return {'document':documents,'search_type':'internal'}
    
    def web_search_node(state:ConditionalState) -> dict:
        '''
        웹 검색(시뮬레이션)
        '''
        mock_result = Document(
            page_content = f"{state['question']}에 대한 검색결과입니다.",
            metadata = {'source':'web'}
        )
        return {'documents' :[mock_result],'search_type':'web'}

    def generate_node(state:ConditionalState) -> dict:
        '''
        답변 생성
        '''
        llm = ChatOpenAI(model='gpt-4o-mini',temperature=0)
        context = '\n'.join([doc.page_content for doc in state['documents']])
        prompt = ChatPromptTemplate.from_template(
            
        )
        chain = prompt | llm | StrOutputParser
        answer = chain.invoke({'context':context,'question':state['question']})
        return {'answer':f'[{state['search_type']}]{answer}'}
    
    # 조건함수
    def decide_search_type(state:ConditionalState) -> Literal['generate','web_search']:
        '''
        검색 결과에 따라 분기
        '''
        if state['documents']:
            return 'generate'
        else :
            return 'web_search'
        
    # 그래프 구축
    graph = StateGraph()
    graph.add_note('iternal_Search',iternal_search_node)
    
    