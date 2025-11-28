import os
import warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    raise ValueError('OPEN_API_KEY not set')

# 필수 라이브러리 로드
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
docs_path = os.path.join(script_dir,'sample_docs')