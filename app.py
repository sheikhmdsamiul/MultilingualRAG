
import warnings
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

from main.pdf_handler import extract_text_from_pdf
from main.clean_processing import preprocess_text, get_vector_store
from main.rag_chatbot import get_response

warnings.filterwarnings("ignore")

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