import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env")
    st.stop()

# 1. Groq LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",   # or any other Groq model
    temperature=0.0,
)


# 2. Prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question **only** using the provided context.
    <context>
    {context}
    </context>

    Question: {input}
    """
)

# 3. Build / cache vector store
@st.cache_resource(show_spinner=False)
def build_vector_store() -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    loader = PyPDFDirectoryLoader("./pdfs")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs[:20])  # limit for demo

    return FAISS.from_documents(chunks, embeddings)

# 4. Streamlit UI
st.set_page_config(page_title="GroqChat with Llama3", page_icon="⚡")
st.title("GroqChat with Llama3")

st.markdown(
    """
**Features**

* Click **Process and Embed PDFs** once to build the vector store.  
* Type a question → get an answer + the source chunks.  
"""
)

#  Button to build vector store 
if st.button("Process and Embed PDFs"):
    with st.spinner("Loading PDFs, chunking & embedding…"):
        start = time.time()
        st.session_state.vectors = build_vector_store()
        elapsed = time.time() - start
    st.session_state.success_msg = f"Vector store ready in **{elapsed:.2f} s**. You can now ask questions."

if "success_msg" in st.session_state:
    st.success(st.session_state.success_msg)

#  Question input 
st.subheader("Ask a question")
user_question = st.text_input("Your question about the PDFs", key="question")

if user_question:
    if "vectors" not in st.session_state:
        st.warning("Please **Process and Embed PDFs** first.")
        st.stop()

    with st.spinner("Retrieving & generating answer…"):
        # 1. Retrieval
        retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 4})

        # 2. Stuff chain
        stuff_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
)

        # 3. Full RAG chain
        rag_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=stuff_chain
        )

        # 4. Invoke the chain
        start = time.time()
        result = rag_chain.invoke({"input": user_question})
        elapsed = time.time() - start

    # Answer display 
    st.subheader("Answer")
    st.write(result["answer"])

    st.info(f"Response generated in **{elapsed:.2f} s**")

    # Source chunks
    with st.expander("Source document segments"):
        for i, doc in enumerate(result.get("context", []), start=1):
            st.markdown(f"**Segment {i}** (page {doc.metadata.get('page', '?')})")
            st.write(doc.page_content)
            st.markdown("---")
