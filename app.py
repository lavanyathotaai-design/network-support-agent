import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

st.set_page_config(page_title="AI Network Support Agent", page_icon="🌐")
st.title("🌐 AI Network Support Agent")
st.write("Ask questions about DNS, SSL, HTTP errors, and connectivity issues.")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = OllamaLLM(model="llama3")

PROMPT = '''
You are a senior network support engineer.
Answer the user's question using the provided context.

Provide the response in this format:

Problem Summary:
Likely Root Cause:
Troubleshooting Steps:
Recommended Next Action:

Context:
{context}

Question:
{question}

Answer:
'''

query = st.text_input("Enter your question")

if query:
    with st.spinner("Analyzing..."):
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        response = llm.invoke(PROMPT.format(context=context, question=query))
        st.subheader("Response")
        st.write(response)

        with st.expander("Relevant Knowledge Base Articles"):
            for doc in docs:
                st.markdown(doc.page_content)
                st.markdown("---")
