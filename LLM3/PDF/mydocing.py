# pip install "docling[all]"
import os
import warnings
warnings.filterwarnings('ignore')

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Docling 변환기
source = "https://arxiv.org/pdf/2408.09869"
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_markdown())