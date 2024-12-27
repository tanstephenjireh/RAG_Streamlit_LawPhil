# 🖼️ RAG (1 LawPhil Doc)

A Retrieval-Augmented Generation (RAG) system using only 1 lawphil case [https://lawphil.net/judjuris/juri2002/jul2002/gr_138726_2002.html]

## 🌟 Key Features

- 🧬 MultiQuery Retriever
- 🤖 OpenAI generative answers
- 📥 Rephrase Query
- 💬 Streaming Responses
- 📄 Reranker 
- 🔍 Document Retrieval
- Returning subscript as sources embedded as a hyperlink
- Streamlit streaming, side by side chat between AI|User

## 🛠️ Technical Stack

- **Embedding Model**: OpenAIEmbeddings
- **LLModel|Rephrase Query**: gpt-4o-mini-2024-07-18
- **Frontend**: Streamlit
- **Vector Database**: Annoy
- **Reranker**: FlashRank
- **ML Framework**: PyTorch

## Future enhancement consideration 
- Login/Signup Feature using Firebase


## ⚡ Quick Start

1. Clone and setup environment:
   ```bash
   git clone 
   python -m venv venv
   source venv/bin/activate  # For Mac/Linux
   # or
   .\venv\Scripts\activate  # For Windows
   pip install -r requirements.txt
   streamlit run streamlit_app.py
   ```


