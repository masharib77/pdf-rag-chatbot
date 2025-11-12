import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Load the API key for Groq
groq_api_key = os.getenv('GROQ_API_KEY')

st.set_page_config(page_title="⚡GroqChat with Llama3⚡", page_icon="⚡")

st.title("⚡GroqChat with Llama3⚡")

st.markdown("""
**🔮Features :**

* 🫵🏻 Click on "Process and Embed PDFs" to prepare the system.
* ⌨️ Enter your question in the input box below.
* 👀 View the answer and related document segments.
""")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only. Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question: {input}
    """
)

def vector_embedding():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    loader = PyPDFDirectoryLoader("./pdfs")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs[:20])
    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors

if st.button("Process and Embed PDFs"):
    with st.spinner("Processing and embedding PDFs... This may take a moment."):
        start_time = time.time()
        st.session_state.vectors = vector_embedding()
        end_time = time.time()
        processing_time = end_time - start_time
    st.session_state.success_message = f"Documents processed and embedded successfully in {processing_time:.2f} seconds. You can now ask questions."

# Display success message if it exists
if 'success_message' in st.session_state:
    st.success(st.session_state.success_message)

st.subheader("Ask Here!🧠")

prompt1 = st.text_input("Enter your question about the documents")

if prompt1 and "vectors" in st.session_state:
    with st.spinner("Generating answer..."):
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start_time = time.time()
        response = retrieval_chain.invoke({'input': prompt1})
        end_time = time.time()
        response_time = end_time - start_time
    
    st.subheader("Answer:")
    st.write(response['answer'])
    
    st.info(f"Response time: {response_time:.2f} seconds")
    
    # With a streamlit expander
    with st.expander("🔎Related Document Segments"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.markdown(f"**Segment {i+1}:**")
            st.write(doc.page_content)
            st.markdown("---")
elif prompt1:
    st.warning("Please process the documents first by clicking the 'Process and Embed PDFs' button.")