from langchain_cohere import CohereRerank
import os

def cohere_reranker():
    compressor_cohere = CohereRerank(model="rerank-english-v2.0", top_n=17, cohere_api_key=os.environ.get("COHERE_API_KEY"))

    return compressor_cohere