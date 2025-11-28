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