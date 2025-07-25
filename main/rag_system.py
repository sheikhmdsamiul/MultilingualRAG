import re
import warnings
import streamlit as st
import os
import tempfile
from bangla_pdf_ocr import process_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

warnings.filterwarnings("ignore")


def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file using OCR"""
    if pdf_file is None:
        return ""
    
    temp_pdf_path = None
    try:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            temp_pdf_path = tmp_file.name
        
        # Extract text using OCR
        extracted_text = process_pdf(temp_pdf_path, language="ben+eng")
        return extracted_text if extracted_text else ""
        
    except Exception as e:
        st.error(f"Error processing PDF file '{pdf_file.name}': {str(e)}")
        return ""
    finally:
        # Ensure the temporary file is deleted after processing
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except OSError:
                pass  # Ignore file deletion errors


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


@st.cache_resource
def get_rerank_model():
    """Load and cache the reranking model"""
    return HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")


def get_retriever_chain(vector_store, groq_chat_instance):
    """Create and return history-aware retriever chain"""
    if not groq_chat_instance or not vector_store:
        raise ValueError("Valid Groq chat instance and vector store required")
    
    try:
        retriever = vector_store.as_retriever(search_kwargs={'k': 10})
        
        rerank_model = get_rerank_model()
        compressor = CrossEncoderReranker(model=rerank_model, top_n=5)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=retriever
        )
        
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", """Based on the above conversation, generate a search query that retrieves the most relevant and up-to-date information for the user. Focus on key topics, entities, or concepts that are directly related to the user's query. 
            Make sure the search query is specific and targets the most relevant sources of information.""")
        ])
        
        history_retriever_chain = create_history_aware_retriever(
            groq_chat_instance, compression_retriever, prompt
        )
        
        return history_retriever_chain
        
    except Exception as e:
        st.error(f"Error creating retriever chain: {str(e)}")
        raise


def get_conversational_rag(history_retriever_chain, groq_chat_instance):
    """Create and return conversational RAG chain"""
    if not history_retriever_chain or not groq_chat_instance:
        raise ValueError("Valid retriever chain and Groq chat instance required")
    
    try:
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a highly knowledgeable multilingual assistant. Your task is to answer user queries using information retrieved from provided PDF documents.
            
            Guidelines:
            - Provide clear and accurate answers based on the retrieved context
            - If the answer is not directly available in the documents, say: "I couldn't find this information in the provided documents."
            - Be concise but thorough in your responses
            - Maintain the language preference of the user (respond in Bangla if asked in Bangla, English if asked in English)
            - Use relevant context from the documents to support your answers
            
            Context from documents:
            {context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])
        
        document_chain = create_stuff_documents_chain(groq_chat_instance, answer_prompt)
        
        conversational_retrieval_chain = create_retrieval_chain(
            history_retriever_chain, document_chain
        )
        
        return conversational_retrieval_chain
        
    except Exception as e:
        st.error(f"Error creating conversational RAG chain: {str(e)}")
        raise


def get_response(user_input, groq_chat_instance):
    """Generate response for user input"""
    if not user_input.strip():
        return "Please provide a valid question."
    
    if "vector_store" not in st.session_state:
        return "Please upload and process PDF files first."
    
    try:
        history_retriever_chain = get_retriever_chain(
            st.session_state.vector_store, groq_chat_instance
        )
        
        conversation_rag_chain = get_conversational_rag(
            history_retriever_chain, groq_chat_instance
        )
        
        response = conversation_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_input
        })
        
        return response.get("answer", "Sorry, I couldn't generate a response.")
        
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "Sorry, there was an error processing your request."


def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False


def main():
    st.set_page_config(
        page_title="AI Assistant📝",
        page_icon="📝",
        layout="wide"
    )
    st.header("AI Assistant📝")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for API Key and PDF uploads
    with st.sidebar:
        st.title("Menu:")
        
        # API Key input
        groq_api_key_input = st.text_input(
            "Enter your Groq API Key:", 
            type="password", 
            key="groq_api_key_input"
        )
        
        # Store API key in session state
        if groq_api_key_input:
            st.session_state.groq_api_key = groq_api_key_input
        
        # Initialize Groq Chat model if API key is available
        groq_chat_instance = None
        if "groq_api_key" in st.session_state and st.session_state.groq_api_key:
            try:
                groq_chat_instance = ChatGroq(
                    groq_api_key=st.session_state.groq_api_key,
                    model_name='llama-3.3-70b-versatile',
                    temperature=0.1  # Lower temperature for more focused responses
                )
                st.sidebar.success("✅ Groq API Key set successfully!")
                
            except Exception as e:
                st.sidebar.error(f"❌ Error initializing Groq: {e}")
                groq_chat_instance = None
        else:
            st.sidebar.warning("⚠️ Please enter your Groq API Key to enable chat.")
        
        # PDF file uploader
        pdf_files = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            type=["pdf"], 
            accept_multiple_files=True
        )
        
        process_clicked = st.button("Submit & Process", type="primary")
        
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.vector_store = None
            st.session_state.pdf_processed = False
            st.rerun()
    
    # Main content area
    if not groq_chat_instance:
        st.warning("Please enter a valid Groq API Key in the sidebar to continue.")
        return
    
    if not pdf_files:
        st.info("📄 Please upload PDF files from the sidebar to get started.")
        return
    
    # Process PDFs
    if process_clicked:
        if not pdf_files:
            st.error("No PDF files uploaded!")
            return
        
        with st.spinner("Processing PDF files..."):
            all_text = ""
            
            # Process each PDF file
            for i, pdf_file in enumerate(pdf_files):
                with st.spinner(f"Processing {pdf_file.name} ({i+1}/{len(pdf_files)})..."):
                    extracted_text = extract_text_from_pdf(pdf_file)
                    if extracted_text:
                        all_text += f"\n\n--- Content from {pdf_file.name} ---\n\n"
                        all_text += extracted_text
                    else:
                        st.warning(f"No text extracted from {pdf_file.name}")
            
            if not all_text.strip():
                st.error("No text was extracted from any of the uploaded PDF files.")
                return
            
            # Preprocess and create documents
            cleaned_text = preprocess_text(all_text)
            if not cleaned_text.strip():
                st.error("No valid content found after text cleaning.")
                return
            
            documents = [Document(page_content=cleaned_text)]
            
            try:
                # Create vector store
                with st.spinner("Creating vector store..."):
                    st.session_state.vector_store = get_vector_store(documents)
                
                # Initialize chat history
                st.session_state.chat_history = [
                    AIMessage(content="Hi! I've processed your PDF files. How can I help you?")
                ]
                st.session_state.pdf_processed = True
                
                st.sidebar.success(f"✅ Successfully processed {len(pdf_files)} PDF file(s)!")
                
            except Exception as e:
                st.error(f"Failed to process PDF files: {str(e)}")
                return
    
    # Chat interface
    if st.session_state.pdf_processed and st.session_state.vector_store:
        # Display chat history
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.write(message.content)
            else:
                with st.chat_message("user"):
                    st.write(message.content)
        
        # User input
        user_input = st.chat_input("Type your message here...")
        if user_input and user_input.strip():
            # Add user message to chat history
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Generate and display AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_response(user_input, groq_chat_instance)
                st.write(response)
                
                # Add AI response to chat history
                st.session_state.chat_history.append(AIMessage(content=response))
    
    elif pdf_files and not st.session_state.pdf_processed:
        st.info("Click 'Submit & Process' to process the uploaded PDF files.")


if __name__ == "__main__":
    main()