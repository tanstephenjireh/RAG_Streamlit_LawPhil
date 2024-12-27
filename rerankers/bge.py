
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

def BgeRerank():
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    bge_compressor = CrossEncoderReranker(model=model, top_n=10)

    return bge_compressor