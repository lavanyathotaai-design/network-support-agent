import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

st.set_page_config(page_title="AI Network Support Agent", page_icon="🌐")
st.title("🌐 AI Network Support Agent")

# Groq client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Load vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

query = st.text_input("Ask a network troubleshooting question")

if query:
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f\"\"\"
You are a senior network support engineer.

Context:
{context}

Question:
{query}

Provide:
- Problem Summary
- Likely Root Cause
- Troubleshooting Steps
- Recommended Next Action
\"\"\"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )

    st.write(response.choices[0].message.content)
