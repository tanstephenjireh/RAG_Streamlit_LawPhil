from typing import List
from langchain_core.output_parsers import BaseOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Annoy
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os

def _load_annoy():
    embedding = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=os.getenv("OPENAI_API_KEY"))
    ANNOY_LOCAL_PATH = "kb/poc_bot_kb"
    annoy_db = Annoy.load_local(
                        ANNOY_LOCAL_PATH, 
                        embeddings=embedding,
                        allow_dangerous_deserialization=True
                    )
    # retriever=annoy_db.as_retriever(search_kwargs={"k": 15})

    return annoy_db
    # return retriever

def multi_query_ret(user_query):

    annoy_db = _load_annoy()

    class LineListOutputParser(BaseOutputParser[List[str]]):
        """Output parser for a list of lines."""

        def parse(self, text: str) -> List[str]:
            lines = text.strip().split("\n")
            return lines


    output_parser = LineListOutputParser()

    temp="""You are an AI language model assistant. Your task is to generate 3
    different versions of the given user question to retrieve relevant documents from a vector
    database. Generate coherent versions of the question asked by the user.
    Original question: {question}"""

    QUERY_PROMPT = ChatPromptTemplate.from_template(temp)

    llm_multi = ChatOpenAI(temperature=0.5, openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Chain
    # llm_chain = LLMChain(llm=llm_multi, prompt=QUERY_PROMPT, output_parser=output_parser)
    llm_chain = QUERY_PROMPT | llm_multi | output_parser

    captured_qs=llm_chain.invoke({
        "question": user_query
    })

    docs_document_object = []
    for question in captured_qs:
        init_vdb = annoy_db.as_retriever(search_kwargs={"k": 15})
        documents=init_vdb.get_relevant_documents(query=question)
        
        for doc in documents:
            docs_document_object.append(doc)

    combine_dop = []
    for doc_object in docs_document_object:
        dop = {}
        dop[doc_object.page_content] = doc_object
        combine_dop.append(dop)

    unique_keys = set()
    result = []

    for d in combine_dop:
        for key, value in d.items():
            if key not in unique_keys:
                unique_keys.add(key)
                result.append({key: value})


    unique_docs = []
    for res in result:
        for key, value in res.items():
            unique_docs.append(value)
    
    return unique_docs