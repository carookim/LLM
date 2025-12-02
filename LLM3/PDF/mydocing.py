# pip install "docling[all]"
import os
import warnings
warnings.filterwarnings('ignore')

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from typing import List, Literal

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
        self.file_parth = file_path
    def load(self) -> List(Document):
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