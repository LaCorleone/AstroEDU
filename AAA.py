import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import openai
import streamlit as st
import warnings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage

# Configura la tua logica di AI
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

vectorstore = Chroma(embedding_function=embeddings,persist_directory="./chroma_db_final/chroma_db_final_new")

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=openai_api_key)

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

qa_system_prompt = """
Comportati come un esperto in didattica. 
Se la domanda è generica per esempio "consigliami qualche attività didattica da fare" oppure "cerco qualcosa" e frasi simili a queste e ti viene chiesto qualcosa senza specificare l'argomento da trattare, allora chiedimi l'argomento, l'età e la durata di quello che sto richiedendo.
Rileva la lingua che viene utilizzata nelle domande ed utilizza la stessa lingua per rispondermi.
Utilizza solo le informazioni che hai per rispondere alle domande e se non hai la risposta dimmi che non lo sai.

Quando viene richiesta una attività specifica, includi sempre il titolo, l’età, il livello, la durata della stessa e il link associato all'attività.

Context: {context}

Answer: 
"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Configura la pagina
#st.set_page_config(page_title="AstroEdu AI Assistant", layout="wide")

# Intestazione
#st.markdown("<h1 style='text-align: center; color: #0004ff;'>A.A.A.: AstroEdu AI Assistant</h1>", unsafe_allow_html=True)
st.image("./LOGO2.webp", use_column_width=True) 

# Sezione di Benvenuto
st.markdown("<h2 style='color: #FFA500;'>Welcome to AstroEDU AI Assistant!</h2>", unsafe_allow_html=True)
st.markdown("I'm here to help you find and make the best use of educational materials from AstroEDU.<br>How can I assist you? If you want, speak to me in your language!", unsafe_allow_html=True)

# Funzione per ottenere la risposta dall'assistente AI
def get_ai_response(question, chat_history):
    response = rag_chain.invoke({
        "chat_history": chat_history,
        "input": question
    })
    return response['answer']

# Funzione per gestire l'invio dei messaggi tramite il campo di input della chat
def chat_actions():
    user_input = st.session_state["chat_input"]
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    
    chat_history = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state["chat_history"]]
    ai_response = get_ai_response(user_input, chat_history)
    
    st.session_state["chat_history"].append({"role": "assistant", "content": ai_response})


# Inizializza la cronologia della chat se non esiste
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Campo di input per i messaggi della chat
st.chat_input("Enter your message", on_submit=chat_actions, key="chat_input")

# Visualizza la cronologia dei messaggi della chat
for i in st.session_state["chat_history"]:
    with st.chat_message(name=i["role"]):
        st.write(i["content"])     
