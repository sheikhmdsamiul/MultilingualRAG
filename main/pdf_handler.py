import os
import streamlit as st
import tempfile
from bangla_pdf_ocr import process_pdf


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