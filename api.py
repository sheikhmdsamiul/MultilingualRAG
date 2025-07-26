from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from main.pdf_handler import extract_text_from_pdf
from main.clean_processing import preprocess_text, get_vector_store
from main.rag_chatbot import get_response, get_retriever_chain, get_conversational_rag
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
import os

app = FastAPI()

# Global state for vector store and chat history
vector_store = None
chat_history = []

class ChatRequest(BaseModel):
    query: str
    api_key: str = None

@app.post("/upload_pdfs")
async def upload_pdfs(files: list[UploadFile] = File(...), api_key: str = Form(...)):
    global vector_store, chat_history
    all_text = ""
    for pdf_file in files:
        content = await pdf_file.read()
        extracted_text = extract_text_from_pdf(content)
        if extracted_text:
            all_text += f"\n\n--- Content from {content.filename} ---\n\n"
            all_text += extracted_text
    cleaned_text = preprocess_text(all_text)
    documents = [Document(page_content=cleaned_text)]
    vector_store = get_vector_store(documents)
    chat_history = [AIMessage(content="Hi! I've processed your PDF files. How can I help you?")]
    return {"message": "PDFs processed and vector store created."}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    global vector_store, chat_history
    groq_api_key = request.api_key or os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return JSONResponse(status_code=400, content={"error": "Groq API Key is required."})
    if vector_store is None:
        return JSONResponse(status_code=400, content={"error": "Please upload and process PDF files first."})
    try:
        groq_chat_instance = ChatGroq(
            groq_api_key=groq_api_key,
            model_name='llama-3.3-70b-versatile',
            temperature=0.1
        )
        # Use the same logic as Streamlit, but with global state
        history_retriever_chain = get_retriever_chain(vector_store, groq_chat_instance)
        conversation_rag_chain = get_conversational_rag(history_retriever_chain, groq_chat_instance)
        response = conversation_rag_chain.invoke({
            "chat_history": chat_history,
            "input": request.query
        })
        answer = response.get("answer", "Sorry, I couldn't generate a response.")
        chat_history.append(HumanMessage(content=request.query))
        chat_history.append(AIMessage(content=answer))
        return {"response": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})