import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder



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