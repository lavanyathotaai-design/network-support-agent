import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

st.set_page_config(page_title="AI Network Support Agent", page_icon="🌐")
st.title("🌐 AI Network Support Agent")
st.write("Ask questions about DNS, SSL, HTTP errors, and connectivity issues.")

# Initialize Groq client
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
    with st.spinner("Analyzing..."):
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a senior network support engineer.

Use the context below to answer the user's question.

Provide the answer in this format:

Problem Summary:
Likely Root Cause:
Troubleshooting Steps:
Recommended Next Action:

Context:
{context}

Question:
{query}
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        st.subheader("Response")
        st.write(response.choices[0].message.content)

        with st.expander("Relevant Knowledge Base Articles"):
            for doc in docs:
                st.markdown(doc.page_content)
                st.markdown("---")
