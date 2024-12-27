import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from retrievers.custom_multiquery import multi_query_ret
from pre_process_query.rephrase_query import rephrase_query
from rerankers.flashrank import flash_rerank
from conversation_chain.custom_chain import conversational_chain
import time
import uuid
import base64

load_dotenv()

# app config
st.set_page_config(page_title="Streamlit Chatbot", page_icon="🤖")
st.title("Law Phil :books:")
st.info('Responses should not be misconstrued as legal advise.', icon="ℹ️")
    

def pq(user_query, chat_history):
    rq = rephrase_query(user_query, chat_history)
    return rq

def doc_to_pass(mqr_ret_docs):
    passage_list = []
    for i, docs in enumerate(mqr_ret_docs):
        passage_dic = {}
        passage_dic['id'] = i+1
        passage_dic['text'] = docs.page_content
        passage_dic['metadata'] = docs.metadata
        passage_list.append(passage_dic)
    return passage_list

def retrieval(user_query):
    # Custom MultiQuery Retrieval
    mqr_ret_docs=multi_query_ret(user_query)

    # Convert Documents to passage list for reranker
    passage_list = doc_to_pass(mqr_ret_docs)

    # Rerank using Flashrank Reranker
    flashrank_ranked_result = flash_rerank(user_query, passage_list)

    contexts = [tx['text']+'\n' for tx in flashrank_ranked_result]

    contexts = "\n".join(contexts)

    return contexts

def get_response(user_query, chat_history):
    rephrase_query = pq(user_query, chat_history)
    # st.write(rephrase_query) check if the rephrasing of query works which is YES
    contexts = retrieval(rephrase_query)
    response = conversational_chain(rephrase_query, contexts)

    return response

def main():

    # sidebar_ui_functionalities()    

    welcome_message = "Hello! What can I help with? 😊"
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content=welcome_message),
        ]

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                # typewriter(text=welcome_message, speed=17)
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    # user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        

        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.markdown(
            """
        <style>
            .st-emotion-cache-janbn0 {
                flex-direction: row-reverse;
                text-align: right;
            }
        </style>
        """,
            unsafe_allow_html=True,
        )
        # st.write(st.session_state.chat_history)
        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            with st.spinner(""):
                response = st.write_stream(get_response(user_query, st.session_state.chat_history))
            # st.write(response)

        st.session_state.chat_history.append(AIMessage(content=response))

        # Remove the added welcome message in the chat history
        if st.session_state.chat_history[0].content == welcome_message:
            del st.session_state.chat_history[0]

        # Only retain the 5 latest conversation like BufferWindowMemory, so if 10 convs replace 12 with 22
        if (len(st.session_state.chat_history)) == 12: # Retain only the latest 5 conversation between user and bot
            del st.session_state.chat_history[0] # Delete the first element
            del st.session_state.chat_history[0] # Delete the following second element which is now positioned to the first
        
        # st.write(st.session_state.chat_history)



if __name__ == "__main__":
    main()