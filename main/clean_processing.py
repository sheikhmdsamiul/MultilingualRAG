import re
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def clean_line(line):
    """Clean individual lines of text"""
    if not line or not line.strip():
        return ""
     
    # Remove page-related lines (English or Bangla)
    line = re.sub(r'(Page|পৃষ্ঠা)\s*\d+', '', line, flags=re.IGNORECASE)
    
    # Remove URLs and emails
    line = re.sub(r'(https?://\S+|www\.\S+|\S+@\S+)', '', line)
    
    # Remove other special characters (keep Bangla punctuations)
    line = re.sub(r'["""\'''•*_+=<>©®@#$%^&~|`]', '', line)
    
    return line.strip()


def preprocess_text(text):
    """Preprocess extracted text by cleaning lines"""
    if not text:
        return ""
    
    lines = text.splitlines()
    cleaned_lines = []
    
    for line in lines:
        cleaned_line = clean_line(line)
        if cleaned_line:  # Only keep non-empty lines
            cleaned_lines.append(cleaned_line)
    
    return "\n".join(cleaned_lines)


@st.cache_resource
def get_embeddings_model():
    """Load and cache the embeddings model"""
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base"
    )


def get_vector_store(documents):
    """Create and return vectorstore for the documents"""
    if not documents or not any(doc.page_content.strip() for doc in documents):
        raise ValueError("No valid documents provided for vector store creation")
    
    try:
        embeddings = get_embeddings_model()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, 
            chunk_overlap=64, 
            separators=["\n\n", "\n", " ", ""]  
        )
        
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            raise ValueError("No chunks created from documents")
        
        vector_store = FAISS.from_documents(chunks, embeddings)

        return vector_store
        
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        raise
