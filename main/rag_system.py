import re
#import unicodedata
import warnings
import streamlit as st
import os
import tempfile
#from gtts import gTTS
from bangla_pdf_ocr import process_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.retrievers import ContextualCompressionRetriever
from sentence_transformers import SentenceTransformer
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

warnings.filterwarnings("ignore")


def extract_text_from_pdf(pdf_file):
    if pdf_file is not None:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            temp_pdf_path = tmp_file.name
        try:
            extracted_text = process_pdf(temp_pdf_path, language="ben+eng") 
            
        except Exception as e:
            st.error(f"Error reading PDF file: {e}")
        finally:
            # Ensure the temporary file is deleted after processing
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
    return extracted_text
         


def clean_line(line):
    # Remove English words
    line = re.sub(r'\b[a-zA-Z]+\b', '', line)

    # Remove numbers 
    line = re.sub(r'\d+', '', line)

    # Remove page-related lines (English or Bangla)
    line = re.sub(r'(Page|পৃষ্ঠা)\s*\d+', '', line)

    # Remove URLs and emails
    line = re.sub(r'(https?://\S+|www\.\S+|\S+@\S+)', '', line)

    # Remove other special characters (keep Bangla punctuations)
    line = re.sub(r'[“”"\'’‘•*_+=<>©®@#$%^&~|`]', '', line)

    return line.strip()


def preprocess_text(text):
    lines = text.splitlines()
    cleaned_lines = [clean_line(line) for line in lines if line.strip()]
    return "\n".join(cleaned_lines)




# Main app
def main():
    st.set_page_config("AI Assistant📝")
    st.header("AI Assistant📝")

    # Sidebar for API Key and PDF uploads
    with st.sidebar:
        st.title("Menu:")
        groq_api_key_input = st.text_input("Enter your Groq API Key:", type="password", key="groq_api_key_input")
        
        # Store API key in session state
        if groq_api_key_input:
            st.session_state.groq_api_key = groq_api_key_input
        
        # Initialize Groq Chat model if API key is available
        groq_chat_instance = None
        if "groq_api_key" in st.session_state and st.session_state.groq_api_key:
            try:
                groq_chat_instance = ChatGroq(
                    groq_api_key=st.session_state.groq_api_key,
                    model_name='llama-3.3-70b-versatile'
                )
                st.sidebar.success("Groq API Key set successfully!")
            except Exception as e:
                st.sidebar.error(f"Error initializing Groq: {e}. Please check your API key.")
                groq_chat_instance = None
        else:
            st.sidebar.warning("Please enter your Groq API Key to enable chat.")
        
        pdf_files = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            type=["pdf"], accept_multiple_files=True)

        process_clicked = st.button("Submit & Process")

    all_text = ""

    if process_clicked and pdf_files:       
        for pdf_file in pdf_files:
            extracted_text = extract_text_from_pdf(pdf_file)
            if extracted_text:
                all_text += extracted_text
    

        

    elif not pdf_files:
        st.info("Please upload PDF files from the sidebar.")
    
    cleaned_text = preprocess_text(all_text)

    st.text_area("Here is the cleaned text:", cleaned_text, height=300)

    st.text_area("Here is the extracted text:", all_text, height=300)


if __name__ == "__main__":
    main()