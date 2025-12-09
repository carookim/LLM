from langchain_redis import RedisVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from config import REDIS_URL, INDEX_NAME
import os


def create_vectorstore(filepath:str = 'sample.txt'):
    # 문서 로드
    path = 'C:\\KIM\\LLM\\LLM3\\history\\sample.txt'
    loader = DirectoryLoader(
        path = path,
        glob = '**/*.txt',
        loader_cls = TextLoader,
        loader_kwargs = {'encoding':'utf-8'},        
    )
    docs = loader.load()
    
    # 청크 분할
    spliter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20,
    separators= ['\n\n','\n','.',' ','']
    )
    chunks = spliter.split_documents(docs)
    
    # 임베딩 모댈 생성 및 초기화
    embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
    
    # vectorstore에 저장
    vectorstore =  RedisVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    redis_url=REDIS_URL,
    index_name=INDEX_NAME
    )
    return vectorstore