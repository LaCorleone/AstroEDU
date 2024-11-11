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

Rileva la lingua della domanda e rispondi sempre nella stessa lingua della domanda, anche se la domanda contiene una sola parola o una frase breve. 

Se la domanda è generica per esempio "consigliami qualche attività didattica da fare" oppure "cerco qualcosa" e frasi simili a queste e ti viene chiesto qualcosa senza specificare l'argomento da trattare, allora chiedimi l'argomento, l'età e la durata di quello che sto richiedendo.

Usa solo le informazioni a disposizione per rispondere alle domande e se qualcosa non la sai non la inventare e dimmi che non lo sai e non hai niente a disposizione dalla tua intelligenza alimentata dalle sole informazioni presenti su astroedu attraverso i link.
Se ti richiedo un’immagine su un argomento specifico, rispondimi dandomi l'immagine o il link associato a quell'immagine.
Se non hai una risposta precisa oppure non hai una attività disponibile, rispondimi che non hai quella specifica informazione o risorsa didattica. 
Quando viene richiesta una attività specifica, rispondi includendo sempre il titolo, l’età, il livello, la durata, la sezione dei materiali.
Fornisci sempre il associato alla risorsa didattica più rilevante riguardante la domanda che ti viene fatta e sceglilo tra questi:
'https://astroedu.iau.org/en/activities/2403/find-the-hidden-rainbows/',
'https://astroedu.iau.org/en/activities/2406/discover-earths-climate-with-a-balloon/',
'https://astroedu.iau.org/en/activities/2405/the-gravity-battle/',
'https://astroedu.iau.org/en/activities/2404/skao-and-the-mysteries-of-invisible-light/',
'https://astroedu.iau.org/en/activities/2402/chasing-the-moon/',
'https://astroedu.iau.org/en/activities/2401/the-sky-at-your-fingertips/',
'https://astroedu.iau.org/en/activities/2312/light-play/',
'https://astroedu.iau.org/en/activities/2304/can-you-find-the-exoplanet/',
'https://astroedu.iau.org/en/activities/2307/how-do-telescopes-work/',
'https://astroedu.iau.org/en/activities/2306/orion-constellation-in-3d/',
'https://astroedu.iau.org/en/activities/2308/the-sun-in-our-box/',
'https://astroedu.iau.org/en/activities/2301/become-a-geo-detective/',
'https://astroedu.iau.org/en/activities/2302/light-in-a-matchbox/',
'https://astroedu.iau.org/en/activities/2305/let-there-be-light-but-not-too-much/',
'https://astroedu.iau.org/en/activities/2303/moving-constellations/',
'https://astroedu.iau.org/en/activity/2205/age-that-crater/',
'https://astroedu.iau.org/en/activities/2203/lets-play-with-powers-of-10/',
'https://astroedu.iau.org/en/activities/2002/misurare-la-velocita-media-di-una-cometa/',
'https://astroedu.iau.org/en/activities/2202/make-your-own-sun/',
'https://astroedu.iau.org/en/activities/2201/hunting-for-spectra/',
'https://astroedu.iau.org/en/activities/2001/driving-on-mars/',
'https://astroedu.iau.org/en/activities/2103/is-the-sun-rotating-follow-the-sunspots/',
'https://astroedu.iau.org/en/activities/2102/reading-the-rainbow/',
'https://astroedu.iau.org/en/activities/2101/one-million-earths-inside-our-sun/',
'https://astroedu.iau.org/en/activities/1801/measure-the-suns-rotation-period/',
'https://astroedu.iau.org/en/activities/1747/dark-matter-and-dark-energy-part-1-discovering-the-main-components-of-the-universe/',
'https://astroedu.iau.org/en/activities/1749/birth-of-a-black-hole/',
'https://astroedu.iau.org/en/activities/1751/hunting-for-black-holes-lower-secondary-level/',
'https://astroedu.iau.org/en/activities/1748/dark-matter-dark-energy-part-2-understanding-the-nature-of-dark-matter-and-dark-energy/',
'https://astroedu.iau.org/en/activities/1648/navigate-like-a-viking-use-the-sun-not-your-phone/',
'https://astroedu.iau.org/en/activities/1624/the-engine-of-life/',
'https://astroedu.iau.org/en/activities/1636/the-big-meltdown/',
'https://astroedu.iau.org/en/activities/1634/transforming-water-into-acid-and-back/',
'https://astroedu.iau.org/en/activities/1630/oceans-as-a-heat-reservoir/',
'https://astroedu.iau.org/en/activities/1628/where-on-earth-am-i/',
'https://astroedu.iau.org/en/activities/1621/valleys-deep-and-mountains-high/',
'https://astroedu.iau.org/en/activities/1618/a-view-from-above/',
'https://astroedu.iau.org/en/activities/1620/the-climate-in-numbers-and-graphs/',
'https://astroedu.iau.org/en/activities/1647/navigating-with-the-kamal-northern-hemisphere/',
'https://astroedu.iau.org/en/activities/1622/big-telescopes-gravity/',
'https://astroedu.iau.org/en/activities/1646/the-quest-for-longitude/',
'https://astroedu.iau.org/en/activities/1619/the-intertropical-convergence-zone/',
'https://astroedu.iau.org/en/activities/1643/country-movers-visualizing-spatial-scales-in-planetary-and-earth-sciences/',
'https://astroedu.iau.org/en/activities/1638/meteoroids-meteors-and-meteorites/',
'https://astroedu.iau.org/en/activities/1641/impact-craters/',
'https://astroedu.iau.org/en/activities/1642/creating-asteroids/',
'https://astroedu.iau.org/en/activities/1644/childrens-planetary-maps-pluto-charon/',
'https://astroedu.iau.org/en/activities/1718/childrens-planetary-maps-titan/',
'https://astroedu.iau.org/en/activities/1719/childrens-planetary-maps-io/',
'https://astroedu.iau.org/en/activities/1720/childrens-planetary-maps-the-moon/',
'https://astroedu.iau.org/en/activities/1721/childrens-planetary-maps-mars/',
'https://astroedu.iau.org/en/activities/1722/childrens-planetary-maps-venus/',
'https://astroedu.iau.org/en/activities/1645/navigation-in-the-ancient-mediterranean-and-beyond/',
'https://astroedu.iau.org/en/activities/1703/the-4-point-backyard-diurnal-parallax-method/',
'https://astroedu.iau.org/en/activities/1616/evening-sky-watching-for-students/',
'https://astroedu.iau.org/en/activities/1535/street-lights-as-standard-candles/',
'https://astroedu.iau.org/en/activities/1609/how-to-travel-on-earth-without-getting-lost/',
'https://astroedu.iau.org/en/activities/1604/seasons-around-the-world/',
'https://astroedu.iau.org/en/activities/1603/investigating-the-atmosphere-air-takes-up-space/',
'https://astroedu.iau.org/en/activities/1602/continental-climate-and-oceanic-climate/',
'https://astroedu.iau.org/en/activities/1617/create-your-own-astro-music/',
'https://astroedu.iau.org/en/activities/1418/star-hats/',
'https://astroedu.iau.org/en/activities/1413/build-your-own-artificial-satellite/',
'https://astroedu.iau.org/en/activities/1615/know-your-planets/',
'https://astroedu.iau.org/en/activities/1614/sun-earth-and-moon-model/',
'https://astroedu.iau.org/en/activities/1613/make-a-star-lantern/',
'https://astroedu.iau.org/en/activities/1612/history-of-the-universe/',
'https://astroedu.iau.org/en/activities/1611/living-in-the-milky-way/',
'https://astroedu.iau.org/en/activities/1610/lets-map-the-earth/',
'https://astroedu.iau.org/en/activities/1608/making-a-sundial/',
'https://astroedu.iau.org/en/activities/1607/what-is-a-constellation/',
'https://astroedu.iau.org/en/activities/1606/what-is-time/',
'https://astroedu.iau.org/en/activities/1605/day-and-night-in-the-world/',
'https://astroedu.iau.org/en/activities/1601/fizzy-balloons-co2-in-school/',
'https://astroedu.iau.org/en/activities/1504/lunar-day/',
'https://astroedu.iau.org/en/activities/1501/how-many-stars-can-you-see-at-night/',
'https://astroedu.iau.org/en/activities/1512/solar-system-model-on-a-city-map/',
'https://astroedu.iau.org/en/activities/1503/suns-shadow/',
'https://astroedu.iau.org/en/activities/1505/solar-system-model/',
'https://astroedu.iau.org/en/activities/1502/lets-break-the-particles/',
'https://astroedu.iau.org/en/activities/1414/astropoetry-writing/',
'https://astroedu.iau.org/en/activities/1412/blue-marble-in-empty-space/',
'https://astroedu.iau.org/en/activities/1411/the-fibre-optic-cable-class/',
'https://astroedu.iau.org/en/activities/1406/meet-our-home-planet-earth/',
'https://astroedu.iau.org/en/activities/1408/meet-our-neighbours-moon/',
'https://astroedu.iau.org/en/activities/1409/build-a-safe-sun-viewer/',
'https://astroedu.iau.org/en/activities/1410/coma-cluster-of-galaxies/',
'https://astroedu.iau.org/en/activities/1404/deadly-moons/',
'https://astroedu.iau.org/en/activities/1403/globe-at-night-activity-guide/',
'https://astroedu.iau.org/en/activities/1402/how-light-pollution-affects-the-stars-magnitude-readers/',
'https://astroedu.iau.org/en/activities/1401/snakes-ladders-game/',
'https://astroedu.iau.org/en/activities/1311/lunar-landscape/',
'http://astroedu.iau.org/en/activities/1310/why-do-we-have-day-and-night/',
'https://astroedu.iau.org/en/activities/1308/meet-our-neighbours-sun/',
'https://astroedu.iau.org/en/activities/1307/glitter-your-milky-way/',
'https://astroedu.iau.org/en/activities/1306/star-in-a-box-advanced/',
'https://astroedu.iau.org/en/activities/1301/counting-sunspots/',
'https://astroedu.iau.org/en/activities/1305/measure-the-solar-diameter/',
'https://astroedu.iau.org/en/activities/1302/star-in-a-box-high-school/',
'https://astroedu.iau.org/en/activities/1304/model-of-a-black-hole/',
'https://astroedu.iau.org/en/activities/1303/design-your-alien/',
'https://astroedu.iau.org/en/activities/1407/levitating-astronaut/',
'https://astroedu.iau.org/en/activities/1309/how-high-is-the-sky/',
'https://astroedu.iau.org/en/activities/2407/touching-the-stars/',
'https://astroedu.iau.org/en/activities/2408/321-time-for-water-rockets/',
'https://astroedu.iau.org/en/activities/measuring-an-exoplanet/',
'https://astroedu.iau.org/en/activities/2410/cubic-planets/'.

Traduci la risposta nella stessa lingua della domanda.

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
