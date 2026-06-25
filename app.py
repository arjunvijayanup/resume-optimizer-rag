import streamlit as st
import os
from llama_index_core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from dotenv import load_dotenv
import tempfile
import shutil
import base64
from pypdf2 import PdfReader
import io

load_dotenv()

def run_rag(documents, query_text:str, job_title: str, job_description: str, 
            embedding_model: str = "models/gemini-embedding-001",
            generative_model: str = "models/gemini-1.5-flash"
            ) -> str:
    """Running RAG using Gemini models for resume optimization"""
    try:
        llm = Gemini(
            model = generative_model,
            api_key = os.getenv("GEMINI_API_KEY")
            )
        embed_model = GeminiEmbedding(
            model_name = embedding_model,
            api_key = os.getenv("GEMINI_API_KEY")
            )
        Settings.llm = llm
        Settings.embed_model = embed_model

        # Analyzing the resume
        analysis_prompt = f"""
        Analyze this resume in detail. Focus on:
        1. Key skills and expertise
        2. Professional experience and achievements
        3. Education and certifications
        4. Notable projects or accomplishments
        5. Career progression and gaps
        
        Provide a concise analysis in bullet points.
        """
        # Vector store initialization (Splitting into chunks)
        index = VectorStoreIndex.from_documents(documents)

        # Gets the 5 most relevant chunks from the resume based on the anlaysis prompt
        resume_analysis = index.as_query_engine(similarity_top_k = 5).query(analysis_prompt)

        # Generating optimization suggestions
        optimization_prompt = f"""
        Based on the resume analysis and job requirements, provide specific, actionable improvements.
        
        Resume Analysis:
        {resume_analysis}
        
        Job Title: {job_title}
        Job Description: {job_description}
        
        Optimization Request: {query_text}
        
        Provide a direct, structured response in this exact format:

        ## Key Findings
        • [2-3 bullet points highlighting main alignment and gaps]

        ## Specific Improvements
        • [3-5 bullet points with concrete suggestions]
        • Each bullet should start with a strong action verb
        • Include specific examples where possible

        ## Action Items
        • [2-3 specific, immediate steps to take]
        • Each item should be clear and implementable

        Keep all points concise and actionable. Do not include any thinking process or analysis.
        """

        # Generating targeted advice
        optimization_suggestions = index.as_query_engine(similarity_top_k = 5).query(optimization_prompt)

        return str(optimization_suggestions)
    
    except Exception as e:
        raise
    pass

def display_pdf(pdf_file):
    """Display PDF in the sidebar"""
    try:
        st.sidebar.subheader("Resume Preview")
        base64_pdf = base64.b64encode(pdf_file.getvalue().decode('utf-8'))
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" type="application/pdf"></iframe>'
        st.sidebar.markdown(pdf_display, unsafe_allow_html = True)

        return True

    except Exception as e:
        st.sidebar.error(f"Error previewing PDF: {str(e)}")
        return False

def main():
    pass
