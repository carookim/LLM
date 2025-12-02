# pip install unstructured
# pip install pi_heif
# pip install unstructured_inference
# pip install pdf2image
# pip install python-poppler
from langchain_community.document_loaders import UnstructuredPDFLoader

loader = UnstructuredPDFLoader(
    "document.pdf",
    mode="elements",  # 요소별 분리
    strategy="hi_res"  # 고해상도 분석
)
documents = loader.load()
print(documents)