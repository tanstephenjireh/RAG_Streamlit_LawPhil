
from langchain.retrievers.document_compressors import FlashrankRerank
from flashrank import Ranker, RerankRequest

# def flasrankRerank():
#     flash_compressor = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2", top_n=10)

#     return flash_compressor


def flash_rerank(user_query, passage_list):
    ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
    rerankrequest = RerankRequest(query=user_query, passages=passage_list)
    rank_results = ranker.rerank(rerankrequest)

    return rank_results