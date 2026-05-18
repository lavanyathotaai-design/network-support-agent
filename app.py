import os
import glob
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Network Support Agent", page_icon="🌐")

st.title("🌐 AI Network Support Agent")
st.write("Ask questions about DNS, SSL, HTTP errors, and connectivity issues.")

# Initialize Groq
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

# Load knowledge base files
@st.cache_data
def load_documents():
    docs = []
    for file in glob.glob("knowledge_base/*.md"):
        with open(file, "r", encoding="utf-8") as f:
            docs.append({
                "file": os.path.basename(file),
                "content": f.read()
            })
    return docs

documents = load_documents()

# Create embeddings
@st.cache_resource
def create_embeddings():
    texts = [doc["content"] for doc in documents]
    embeddings = model.encode(texts)
    return embeddings

doc_embeddings = create_embeddings()

# Retrieve most relevant documents
def retrieve_docs(query, top_k=3):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [documents[i] for i in top_indices]

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Analyzing..."):
        relevant_docs = retrieve_docs(query)
        context = "\n\n".join(doc["content"] for doc in relevant_docs)

        prompt = f"""
You are a senior network support engineer.

Answer the user's question using the provided context.

Provide the response in the following format:

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
            messages=[{"role": "user", "content": prompt}],
        )

        answer = response.choices[0].message.content

        st.subheader("Response")
        st.write(answer)

        with st.expander("Relevant Knowledge Base Articles"):
            for doc in relevant_docs:
                st.markdown(f"### {doc['file']}")
                st.markdown(doc["content"])
                st.markdown("---")
