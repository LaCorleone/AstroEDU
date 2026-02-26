import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="AstroEDU AI Assistant", layout="wide")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY. Add it in Streamlit Secrets or as an environment variable.")
    st.stop()

PERSIST_DIR = "./chroma_db_final" 
LOGO_PATH = "./LOGO2.webp"


# -----------------------------
# VECTORSTORE + RETRIEVER
# -----------------------------
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=OPENAI_API_KEY
)


# -----------------------------
# PROMPTS
# -----------------------------
contextualize_q_system_prompt = """Given a chat history and the latest user question
which might reference context in the chat history, formulate a standalone question
which can be understood without the chat history. Do NOT answer the question,
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
You are an educational assistant for AstroEDU.

Rules:
1) If the user request is generic (e.g. "recommend some activities" / "I’m looking for something")
   ask 3 questions before answering:
   - topic/subject
   - age (or school level)
   - duration
2) When you recommend activities, ALWAYS output a bullet list.
   For each activity include these fields (if available in the retrieved context/metadata):
   - link
   - age
   - level
   - duration
   - materials
3) After the list, add a short summary of each suggested activity.
4) Reply in the same language used by the user.

Use ONLY the retrieved context below. If the context does not contain enough information,
say what is missing and ask a follow-up question.

Context:
{context}
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


# -----------------------------
# UI
# -----------------------------
if os.path.exists(LOGO_PATH):
    st.image(LOGO_PATH, use_container_width=True)

st.markdown("<h2 style='color: #FFA500;'>Welcome to AstroEDU AI Assistant!</h2>", unsafe_allow_html=True)
st.markdown(
    "I'm here to help you find and use educational materials from AstroEDU.<br>"
    "Ask me in your language if you want!",
    unsafe_allow_html=True
)


# -----------------------------
# CHAT LOGIC
# -----------------------------
def get_ai_response(question: str, chat_history):
    response = rag_chain.invoke({
        "chat_history": chat_history,
        "input": question
    })
    return response.get("answer", "")


if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_msg = st.chat_input("Enter your message")

if user_msg:
    st.session_state["chat_history"].append({"role": "user", "content": user_msg})

    chat_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state["chat_history"]]
    ai_response = get_ai_response(user_msg, chat_history)

    st.session_state["chat_history"].append({"role": "assistant", "content": ai_response})

for m in st.session_state["chat_history"]:
    with st.chat_message(m["role"]):
        st.write(m["content"])