from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os

def conversational_chain(rephrased_query, context):
    template = """As an AI assisstant, your task is to provide answers for philippines laws from the provided context. Refrain from answering OUTSIDE OF THE CONTEXT BELOW.

    If the question is OUT-OF-CONTEXT. Remind the user that you are only answering about Philippine Law concerns.

    BE HONEST if you are not sure or do not know the answer, advise to consult with legal
    experts or the relevant authorities to ensure compliance with the respective laws.

    Feel free to structure your responses having an introduction, body, and conclusion if needed. Else, make it short and straight to the point.

    Always follow this format in answering:
    According to <Document Title> ...

    Moreover, for every paragraph/claim put a subscript like [1], [2], and so on. Put a clickable link in the subscript, use the url provided.

    {context}

    User: {question}
    AI: """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),
                    model_name='gpt-4o-mini-2024-07-18', #gpt-4o-mini-2024-07-18
                    temperature=0.0
                    )
    
    conv_chain = prompt | llm | StrOutputParser()

    return conv_chain.stream({
        "context": context,
        "question": rephrased_query,
    })


    # Moreover, you should be able to give a subscript after a statement or paragraph like [n] and so on as a relevant source where you got the claim.
    # I want a clickable hyperlink embedded to the subscript [n] that directs the user to a url. The link should be embedded in the HTML anchor (<a>) tag, 
    # but make sure the output is in valid HTML syntax.