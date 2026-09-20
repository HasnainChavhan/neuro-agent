import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Neuro-Agent RAG", page_icon="🧠", layout="wide")
st.title("🧠 Neuro-Agent: Document RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for controls and stats
with st.sidebar:
    st.header("Document Management")
    uploaded_file = st.file_uploader("Upload a document (PDF, TXT)", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        if st.button("Process & Index Document"):
            with st.spinner("Processing..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                response = requests.post(f"{API_URL}/upload", files=files)
                if response.status_code == 200:
                    st.success(f"Successfully processed {uploaded_file.name}")
                else:
                    st.error("Failed to process document")
    
    st.divider()
    if st.button("Clear Vector DB"):
        res = requests.delete(f"{API_URL}/reset")
        if res.status_code == 200:
            st.success("Database cleared!")
            
    st.divider()
    st.subheader("Stats")
    try:
        stats_res = requests.get(f"{API_URL}/stats")
        if stats_res.status_code == 200:
            count = stats_res.json()["document_count"]
            st.metric("Total Document Chunks", count)
    except:
        st.metric("API Status", "Offline")

# Chat interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources") and len(msg["sources"]) > 0:
            with st.expander("View Sources"):
                for idx, src in enumerate(msg["sources"]):
                    st.text(f"Source {idx+1}:\n{src}")

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = requests.post(f"{API_URL}/chat", json={"question": prompt})
                if res.status_code == 200:
                    data = res.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    st.markdown(answer)
                    if sources:
                        with st.expander("View Sources"):
                            for idx, src in enumerate(sources):
                                st.text(f"Source {idx+1}:\n{src}")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
