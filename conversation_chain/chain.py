
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os


def conversation_chain(memory, compression_retriever):
    template = """As an AI assisstant, your task is to provide answers for philippines laws from the provided context. Refrain from answering OUTSIDE OF THE CONTEXT BELOW.

    If the question is OUT-OF-CONTEXT. Remind the user that you are only answering about Philippine Law concerns.

    BE HONEST if you are not sure or do not know the answer, advise to consult with legal
    experts or the relevant authorities to ensure compliance with the respective laws.

    Feel free to structure your responses having an introduction, body, and conclusion if needed. Else, make it short and straight to the point.

    Always follow this format in answering:
    (According to <Document Title>)

    Moreover, you should able to give relevant sources at the end when it's needed for your answer. Do not fabricate references and always follow this format:
    Source
    ----------------
    The format should be like this, I want it to be a clickable link using the Document Title as display and its associated URL, either use any of these format:
    <a href="[url]">[Document Title] </a>

    {context}

    User: {question}
    AI: """

    prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template=template
    )

    CONDENSE_QUESTION_TEMPLATE = """
    Given the provided conversation ("CHAT HISTORY") between a "Human" and an "Assistant" and a FOLLOW-UP QUESTION, rephrase the FOLLOW-UP QUESTION into a STANDALONE QUESTION in its original language only if it is related to the conversation. If the FOLLOW-UP QUESTION is unrelated to the CHAT_HISTORY, return the FOLLOW-UP QUESTION.

    CHAT HISTORY:

    {chat_history}

    FOLLOW-UP QUESTION: {question}

    STANDALONE QUESTION:

    """

    condense_question_prompt = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)


    llm = ChatOpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"),
                    model_name='gpt-4o-mini-2024-07-18', #gpt-4o-mini-2024-07-18
                    temperature=0.0,
                    streaming=True,
                    callback_manager=[StreamingStdOutCallbackHandler()]
                    )
    
    combine_docs_chain_kwargs = {"prompt": prompt_template}
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=compression_retriever,
        combine_docs_chain_kwargs=combine_docs_chain_kwargs,
        condense_question_prompt=condense_question_prompt,
        memory=memory
    )

    return conversation_chain