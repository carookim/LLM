# pip install "docling[all]"
import os
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
load_dotenv()

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from typing import List, Literal

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings,ChatOpenAI

## 1
##
# source = "https://arxiv.org/pdf/2408.09869" # document per local path or URL
# # Docling 변환기 생성 : 도구 생성
# converter = DocumentConverter()
# # pdf -> Docling Document 변환
# result = converter.convert(source)
# print(result.document.export_to_markdown()) # output : "## Docling Tachnical Report[...]"

## 2
# # Docling 변환기 생성 : 도구 생성
# converter = DocumentConverter()

# # pdf -> Docling Document 변환
# file_path = r'C:\KIM\LLM\LLM3\PDF\document_table.pdf'
# result = converter.convert(file_path)

# # markdown 추출(표 구조 보존)
# mark_down_content = result.document.export_to_markdown()
# print(mark_down_content)


## 3
class DoclingPDFLoader:
    '''Docling을 사용한 PDF 로더'''
    def __init__(self,file_path:str):
        self.file_path = file_path
    def load(self) -> List[Document]:
        '''PDF를 로드하고 Document 리스트로 반환'''
        converter = DocumentConverter()
        result = converter.convert(self.file_path)
        markdown_content = result.document.export_to_markdown()
        #langchain의 Document 형식으로 반환
        documents = [
            Document(
                page_content=markdown_content,
                metadata={
                    'source':self.file_path,
                    'loader':'docling',
                    'format':'markdown'
                }
            )
        ]
        return documents

## 4
# PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
class SimplePDFLoader:
    '''기본 PDF로더 간단한 텍스트 추출에 적합
    표구조는 보존되지않음
    이미지의 텍스트는 잘안됨'''
    def __init__(self,file_path:str):
        self.file_path = file_path
    def load(self)-> List[Document]:
        '''PDF 로더 (텍스트만 추출)'''
        loader = PyPDFLoader(self.file_path)
        documents = loader.load()
        # 메타데이터에 로더 정보 추가
        for doc in documents:
            doc.metadata['loader'] = 'pypdf'
        return documents


# 스플리터
korean_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100,
    separators=[
        '\n##', # 마크다운  2단계 헤더
        "\n##", # 마크 다운 3단계 헤더
        '\n\n',
        '\n',
        '다',
        '요',
        '니다',
        ' ',
        ''
    ],
    length_function = len,
    is_separator_regex=False
)

## Step 1. 일반적인 chain을 이용한 RAG
# 문서로딩
file_path = r'C:\KIM\LLM\LLM3\PDF\pdf_doc01.pdf'
loader = DoclingPDFLoader(file_path)
docs =loader.load()
print(f'docs : {docs}')

# 청킹
text_spliter =  RecursiveCharacterTextSplitter(file_path)
doc_splits = korean_splitter.split_documents(docs)
print(f'청킹 수 : {len(doc_splits)}')

#  벡터DB
vectorstores = Chroma.from_documents(
    documents=doc_splits,
    embedding=OpenAIEmbeddings(model='text-embedding-3-small'),
    collection_name='crag_collection'
)

# 리트리버
retriever = vectorstores.as_retriever(search_kwargs = {'k':3})

question = '실제 교통 정체상황에서 상호 간섭에 대해서 알려줘'
# 사용자 질문에 대한 리트리버를 수행 context
documents = retriever.invoke('question')
print(f'리트리버가 찾은 context 수 : {len(documents)}')
context = '\n\n---\n\n'.join( doc.page_content for doc in documents)
# context로 LLM을 위한 프폼프트 작성
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template('''
사용자의 질문에 대한 답을 주어진 context 에서만 찾고 해당 사항이 없으면 관련 없음이라고 출력할것
context : 
{context}


사용자 질문 : 
{question}
                                          
출력:
''')
# LLM정의
llm = ChatOpenAI(model='gpt-4o-mini',temperature=0)
# 체인
from langchain_core.output_parsers import StrOutputParser
chain = prompt | llm | StrOutputParser()
# 실행
result = chain.invoke({"context":context, "question":question})
print(f'LLM이 찾은 정답 : {result}')



# step2  랭그래프를 이용한 RAG