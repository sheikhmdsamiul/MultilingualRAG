# MultilingualRAG - Document Question Answering System

A Multilingual Retrieval-Augmented Generation (RAG) system that enables question-answering about PDF documents in Bengali and English using Groq LLM.

## Setup Guide

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/sheikhmdsamiul/MultilingualRAG.git
cd MultilingualRAG
```

2. **Create virtual environment:**
```bash
python -m venv ragenv
ragenv\Scripts\activate  # On Windows
# source ragenv/bin/activate  # On Linux/Mac
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```
**For Bangla PDF OCR installation**
- Install the package:
   ```bash
   pip install bangla-pdf-ocr
   ```

- Run the setup command to install dependencies:
   ```bash
   bangla-pdf-ocr-setup
   ```

4. **Set up Groq API key:**
   - Get your key from https://console.groq.com/
   - Set as environment variable: `set GROQ_API_KEY=your_key` (Windows) or `export GROQ_API_KEY=your_key` (Linux/Mac)
   - Or provide it directly in the web interface

## Usage

### Web Interface (Streamlit)
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser

## Tools, Libraries & Packages Used

### Core Libraries
- **langchain & langchain-groq**: For LLM operations and Groq integration
- **streamlit**: Web interface
- **faiss-cpu**: Vector similarity search

### Text Processing & Retrieval
- **bangla-pdf-ocr**: Bengali-English PDF text extraction
- **sentence-transformers**: For text embeddings
- **HuggingFace models**: For reranking and embeddings

### ML Models
- **intfloat/multilingual-e5-base**: Multilingual embedding model
- **BAAI/bge-reranker-base**: Cross-encoder for reranking
- **llama-3.3-70b-versatile**: Groq's LLM

## Sample Queries and Outputs

### English
**Query:** Who is said to be a good man in the language of Anupam?
**Output:**
> According to the text, Shyamtu Nath is said to be a good man in the language of Anupam.
### Bengali
**Query:** অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?
**Output:**
> অনুপমের ভাষায় সুপুরুষ হলেন শব্তুনাথ। 

**Query:** কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
**Output:**
> অনুপমের মামাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে।

**Query:** বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?
**Output:**
> বিয়ের সময় কল্যাণীর প্রকৃত বয়স ১৫ বছর ছিল।

## Technical Implementation Details

### 1. Text Extraction Method
- Used `bangla-pdf-ocr` library for its specialized Bengali-English OCR capabilities
- Handled formatting challenges:
  - Removed page numbers/headers using regex
  - Cleaned special characters while preserving Bengali punctuation
  - Managed inconsistent line breaks and mixed scripts

### 2. Chunking Strategy
- Used `RecursiveCharacterTextSplitter` with:
  - Chunk size: 512 characters
  - Overlap: 64 characters
  - Separators: `["\n\n", "\n", " ", ""]`
- Works well because it maintains semantic coherence and handles both languages effectively

### 3. Embedding Model
- Using `intfloat/multilingual-e5-base`
- Chosen for excellent Bengali-English performance
- Creates 768-dimensional vectors capturing semantic meaning
- Maintains cross-lingual semantic relationships

### 4. Query-Document Comparison
- Using FAISS for vector similarity search
- Cosine similarity for semantic matching
- Efficient for large-scale retrieval
- Fast and memory-efficient implementation

### 5. Meaningful Comparison
- Two-stage retrieval:
  1. History-aware retriever using chat context
  2. Cross-encoder reranking for better relevance
- Handles vague queries through conversation history
- Returns "Information not found" for out-of-scope queries

### 6. Results and Potential Improvements
Current implementation works well for:
- Direct questions
- Multilingual queries
- Context-aware conversations

Potential improvements:
- Semantic chunking instead of character-based
- Domain-specific embedding fine-tuning
- Enhanced conversation history handling
- Larger context windows for complex queries

## Project Structure
```
MultilingualRAG/
├── app.py                   # Streamlit interface
├── api.py                   # API implementation 
├── main/
│   ├── pdf_handler.py       # PDF processing
│   ├── clean_processing.py  # Text cleaning
│   └── rag_chatbot.py      # Core RAG logic
└── requirements.txt
```

## Note on API Implementation
An attempt has been made to implement a FastAPI REST API but it's currently not functional and needs further development.

## Troubleshooting
- **API Key Errors:** Verify at https://console.groq.com/
