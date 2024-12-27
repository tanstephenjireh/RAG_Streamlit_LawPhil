from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os

def rephrase_query(user_query, chat_history):
    CONDENSE_QUESTION_TEMPLATE = """
    Given the provided conversation ("CHAT HISTORY") between a "Human" and an "Assistant" and a FOLLOW-UP QUESTION, rephrase the FOLLOW-UP QUESTION into a STANDALONE QUESTION in its original language only if it is related to the conversation. If the FOLLOW-UP QUESTION is unrelated to the CHAT_HISTORY, return the FOLLOW-UP QUESTION.

    CHAT HISTORY:

    {chat_history}

    FOLLOW-UP QUESTION: {question}

    STANDALONE QUESTION:

    """

    prompt = ChatPromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)

    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),
                    model_name='gpt-4o-mini-2024-07-18', #gpt-4o-mini-2024-07-18
                    temperature=0.0
                    )
    rq_chain = prompt | llm | StrOutputParser()
    
    return rq_chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })