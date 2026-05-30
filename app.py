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

# Retrieve most relevant documents (uses latest query only for best RAG accuracy)
def retrieve_docs(query, top_k=3):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [documents[i] for i in top_indices]

# ── STEP 1: Initialise session state (guard prevents reset on rerun) ──
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── STEP 2: Display full conversation history ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── STEP 3: Chat input box ──
query = st.chat_input("Ask a question about DNS, SSL, HTTP errors...")

if query:
    # Show and store user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Analyzing..."):
        # Retrieve relevant docs based on latest query only
        relevant_docs = retrieve_docs(query)
        context = "\n\n".join(doc["content"] for doc in relevant_docs)

        # ── STEP 4: Build system prompt ──
        system_prompt = f"""You are a senior network support engineer.
Answer the user's question using the provided context.
Always provide your response in this format:

Problem Summary:
Likely Root Cause:
Troubleshooting Steps:
Recommended Next Action:

Context:
{context}"""

        # ── STEP 5: Include last 5 turns of conversation history ──
        history = st.session_state.messages[-10:]  # last 5 user+assistant pairs
        api_messages = [{"role": "system", "content": system_prompt}] + [
            {"role": m["role"], "content": m["content"]}
            for m in history
        ]

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=api_messages,
        )

        answer = response.choices[0].message.content

    # Show and store assistant message
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Show relevant knowledge base articles
    with st.expander("📚 Relevant Knowledge Base Articles"):
        for doc in relevant_docs:
            st.markdown(f"### {doc['file']}")
            st.markdown(doc["content"])
            st.markdown("---")
