
import os
from dotenv import load_dotenv
from langchain_community.retrievers import TavilySearchAPIRetriever
load_dotenv()
os.environ.get('TAVILY_API_KEY')


retriever = TavilySearchAPIRetriever()
result = retriever.invoke('2025 멜론 뮤직 어워드 시상 내역은 어떻게 되나요')
context = '\n\n--\n\n'.json([doc.page_content for doc in result])
print(context)