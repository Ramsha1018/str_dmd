from utils import *
import streamlit as st

st.title("Welcome to Duchenne Muscular Dystrophy Chatbot")
st.write("This chatbot is about Duchenne Muscular Dystrophy (DMD). You can ask any question from this chatbot and we will respond to you. DMD is a severe, progressive genetic disorder that primarily affects young boys, occurring in about 1 in 3,500 male births. For more information you can ask.")

try:
    vectordb = load_local_vectordb_using_qdrant()
except Exception as e:
    st.error(f"Failed to load vector database: {e}")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'source_url' not in st.session_state:
    st.session_state.source_url = ""
if 'show_url' not in st.session_state:
    st.session_state.show_url = False

# Display the chat history
for chat in st.session_state.chat_history:
    st.write(f"**You:** {chat['query']}")
    st.write(f"**Bot:** {chat['response']}")

query = st.text_input("Ask Anything About DMD")

if query:
    try:
        response, source_url = retri_answer(query, vectordb, st.session_state.chat_history)
        st.session_state.chat_history.append({'query': query, 'response': response})
        st.write(f"**You:** {query}")
        st.write(f"**Bot:** {response}")
        st.session_state.source_url = source_url
        st.session_state.show_url = True
    except Exception as e:
        st.error(f"Error retrieving answer: {e}")

if st.session_state.show_url and st.button("Show URL"):
    st.write(st.session_state.source_url)
