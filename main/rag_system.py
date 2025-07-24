import re
import unicodedata
import warnings
import streamlit as st
import os
#from gtts import gTTS
import fitz  # PyMuPDF
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

warnings.filterwarnings("ignore")


def extract_text_from_pdf(pdf_file):
    if pdf_file is not None:
        try:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            full_text = ""
            for page in doc:
                text = page.get_text("text")  # Use "text" mode for natural layout
                full_text += text + "\n"
    
            doc.close()
            return full_text
        except Exception as e:
            st.error(f"Error reading PDF file: {e}")
            return ""
         
        
    return ""



def clean_line(line):
    #Remove extra spaces
    line = re.sub(r'\s+', ' ', line.strip())

    #Bengali OCR fixes
    ocr_fixes = {
        'িি': 'ি', 'েে': 'ে', '্্': '্', 'োো': 'ো', 'ুু': 'ু',
        'াা': 'া', 'ীী': 'ী', 'ুূ': 'ূ', 'েে': 'ে'
    }
    for wrong, right in ocr_fixes.items():
        line = line.replace(wrong, right)

    #Replace English punctuation in Bengali context
    line = re.sub(r'\.(?=\s*[অ-হ])', '।', line)

    #Keep useful characters
    line = re.sub(r'[^\w\s\u0980-\u09FF\u0900-\u097F.,!?;:()\-"\'।]', ' ', line)

    #Normalize spacing around punctuation
    line = re.sub(r'([.!?।])([^\s])', r'\1 \2', line)
    line = re.sub(r'\s+([.!?।,;:])', r'\1', line)

    #Reduce repeated punctuation
    line = re.sub(r'[.]{2,}', '.', line)
    line = re.sub(r'[।]{2,}', '।', line)

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