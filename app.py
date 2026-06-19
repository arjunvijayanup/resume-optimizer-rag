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

    pass

def display_pdf(pdf_file):
    """Display PDF in the sidebar"""

    pass

def main():
    pass
