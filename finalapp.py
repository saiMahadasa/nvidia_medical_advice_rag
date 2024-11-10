import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv

load_dotenv()

# Load the Groq API key
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

# Vector embedding function
def vector_embedding():
    if "vectors" not in st.session_state:  # Check if vectors exist
        try:
            # Initialize embeddings and document loader
            st.session_state.embeddings = NVIDIAEmbeddings()
            st.session_state.loader = PyPDFDirectoryLoader("./medical_data")  # Load your medical data folder

            # Load documents from the folder
            st.session_state.docs = st.session_state.loader.load()

            # Check if there are enough documents
            if len(st.session_state.docs) == 0:
                st.error("No documents found in the specified folder. Please add medical data to './medical_data'.")
                return

            # Set up the text splitter
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
            
            # Only take up to 30 documents if available
            documents_to_process = st.session_state.docs[:30]
            if len(documents_to_process) == 0:
                st.error("The documents in the folder are too few or empty. Please add more documents.")
                return
            
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(documents_to_process)

            print("Embedding process started...")
            # Create the vector store using the documents and embeddings
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            st.write("Medical Vector Store DB is Ready!")
        except Exception as e:
            st.error(f"Error during embedding process: {e}")

# Streamlit UI setup
st.title("Health Symptoms & Medical Advice Demo")
llm = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")

# Define the new prompt template (updated without `output` variable)
from langchain.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(
    """
    Based on the provided medical information, please answer the following health-related question.
    Provide an accurate diagnosis or medical advice based only on the context of the symptoms mentioned.
    <context>
    {context}
    </context>
    User's Symptoms: {input}
    Disease Diagnosis and Medical Advice: 
    """
)

prompt1 = st.text_input("Enter Your Health Symptoms (e.g., fever, cough, fatigue, etc.)")

# Button to trigger embedding
if st.button("Start Document Embedding"):
    vector_embedding()  # Initialize vectors

# Process the user's question based on symptoms
if prompt1:
    # Make sure the vectors exist before using them
    if "vectors" in st.session_state:
        try:
            # Create document chain using the ChatPromptTemplate
            document_chain = create_stuff_documents_chain(llm, prompt)

            # Create the retriever and retrieval chain
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start = time.process_time()
            # Pass the input to the retrieval chain
            response = retrieval_chain.invoke({'input': prompt1})
            st.write("Response time:", time.process_time() - start)
            st.write("Medical Advice & Disease Diagnosis: ", response['answer'])

            # Display document similarity search results (context)
            with st.expander("Related Medical Information"):
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
        except Exception as e:
            st.error(f"Error during retrieval process: {e}")
    else:
        st.write("Please run the 'Start Document Embedding' process first.")
