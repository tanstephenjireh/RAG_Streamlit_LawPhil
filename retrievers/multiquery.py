from typing import List
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import BaseOutputParser

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Annoy
import streamlit as st

import os
from dotenv import load_dotenv
load_dotenv()

def _load_annoy():
    embedding = OpenAIEmbeddings(model='text-embedding-ada-002', api_key=st.secrets["OPENAI_API_KEY"])
    ANNOY_LOCAL_PATH = "kb/poc_bot_kb"
    annoy_db = Annoy.load_local(
                        ANNOY_LOCAL_PATH, 
                        embeddings=embedding,
                        allow_dangerous_deserialization=True
                    )
    retriever=annoy_db.as_retriever(search_kwargs={"k": 20})

    # return annoy_db
    return retriever

def multi_query_ret():

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

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template=temp,
    )
    llm_multi = ChatOpenAI(temperature=0.5, api_key=st.secrets["OPENAI_API_KEY"])

    # Chain
    # llm_chain = LLMChain(llm=llm_multi, prompt=QUERY_PROMPT, output_parser=output_parser)
    llm_chain = QUERY_PROMPT | llm_multi | output_parser

    retriever_laws = MultiQueryRetriever(
        retriever=annoy_db.as_retriever(search_kwargs={"k": 20}), llm_chain=llm_chain, parser_key="lines"
    )  # "lines" is the key (attribute name) of the parsed output

    return retriever_laws