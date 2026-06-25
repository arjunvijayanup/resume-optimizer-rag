import streamlit as st
import os
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
import tempfile
import shutil
import base64
import io

load_dotenv()

def run_rag(documents, query_text:str, job_title: str, job_description: str, 
            embedding_model: str = "BAAI/bge-base-en-v1.5",
            generative_model: str = "llama-3.1-8b-instant"
            ) -> str:
    """Running RAG using Gemini models for resume optimization"""
    try:
        llm = Groq(
            model = generative_model,
            api_key = os.getenv("GROQ_API_KEY")
            )
        embed_model = HuggingFaceInferenceAPIEmbedding(
            model_name = embedding_model
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
        base64_pdf = base64.b64encode(pdf_file.getvalue()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" type="application/pdf"></iframe>'
        st.sidebar.markdown(pdf_display, unsafe_allow_html = True)

        return True

    except Exception as e:
        st.sidebar.error(f"Error previewing PDF: {str(e)}")
        return False

def main():
    st.set_page_config(page_title = "Resume Optimizer", layout = "wide")

    # Initializing session states
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "docs_loaded" not in st.session_state:
        st.session_state.docs_loaded = False
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = None
    if "current_pdf" not in st.session_state:
        st.session_state.current_pdf = None

    # Header
    st.title("Resume Optimizer")
    st.caption("Powered by Groq + HuggingFace")

    # Sidebar for configuration
    with st.sidebar:
        
        # Model selection
        generative_model = st.selectbox(
            "Generative Model",
            [
                "llama-3.1-8b-instant",
                "llama-3.3-70b-versatile",
                "qwen-qwq-32b"
            
            ]
        )

        embedding_model = st.selectbox(
            "Embedding Model",
            [  
                "BAAI/bge-small-en-v1.5",
                "BAAI/bge-base-en-v1.5",
                "BAAI/bge-m3"
            
            ]
        )

        st.divider()

        # Resume upload button
        st.subheader("Upload Resume")
        uploaded_file = st.file_uploader(
            "Choose you resume (PDF)",
            type = "pdf",
            accept_multiple_files = False
        )

        # PDF upload and processing handler
        if uploaded_file is not None:
            if uploaded_file != st.session_state.current_pdf: # If uploaded pdf is same as already present pdf in session state
                st.session_state.current_pdf = uploaded_file
                try:
                    if not os.getenv("GROQ_API_KEY"):
                        st.error("Missing API Key")
                        st.stop()

                    # Creating temp directory for the pdf
                    if st.session_state.temp_dir:
                        shutil.rmtree(st.session_state.temp_dir) # Removing dir if already temp dir present
                    st.session_state.temp_dir = tempfile.mkdtemp() # Create temp dir

                    # Saving uploaded PDF to temp directory
                    file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f: # Opening in write, binary mode
                        f.write(uploaded_file.getbuffer()) # extracting raw bytes from uploaded pdf and written to the disk
                    
                    with st.spinner("Loading Resume..."):
                        documents = SimpleDirectoryReader(st.session_state.temp_dir).load_data() # Reading everything in the temp folder
                        st.session_state.docs_loaded = True
                        st.session_state.documents = documents
                        st.success("Resume Loaded Successfully")
                        display_pdf(uploaded_file)
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Job Information")
        job_title = st.text_input("Job Title")
        job_description = st.text_area("Job Description", height = 200)

        st.subheader("Optimization Options")
        optimization_type = st.selectbox("Select Optimization Type",
                                         [
                                             "ATS Keyword Optimizer",
                                             "Experience Section Enhancer",
                                             "Skills Hierarchy Creator",
                                             "Professional Summary Crafter",
                                             "Education Optimizer",
                                             "Technical Skills Showcase",
                                             "Career Gap Framing"
                                         ])
        
        if st.button("Optimize Resume"):

            if not st.session_state.docs_loaded:
                st.error("Please upload your resume first")
                st.stop()
            if not job_title or not job_description:
                st.error("Please provide both Job title and Job description")
                st.stop()
            
            # Optimization prompts based on user optimization type selections
            prompts = {
                "ATS Keyword Optimizer": "Identify and optimize ATS keywords. Focus on exact matches and semantic variations from the job description.",
                "Experience Section Enhancer": "Enhance experience section to align with job requirements. Focus on quantifiable achievements.",
                "Skills Hierarchy Creator": "Organize skills based on job requirements. Identify gaps and development opportunities.",
                "Professional Summary Crafter": "Create a targeted professional summary highlighting relevant experience and skills.",
                "Education Optimizer": "Optimize education section to emphasize relevant qualifications for this position.",
                "Technical Skills Showcase": "Organize technical skills based on job requirements. Highlight key competencies.",
                "Career Gap Framing": "Address career gaps professionally. Focus on growth and relevant experience."
            }

            with st.spinner("Analyzing resume and generating suggestions..."):
                try:
                    response = run_rag(
                        st.session_state.documents,
                        prompts[optimization_type],
                        job_title,
                        job_description,
                        embedding_model,
                        generative_model
                    )
                    # Remvoing thinking tags from the responses
                    #response = response.replace("<think>", "").replace("</think>", "")
                    st.session_state.messages = [{"role": "assistant", "content": response}]
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
            st.divider()
    
    with col2:
        st.subheader("Optimization Results")
        for message in st.session_state.messages:
            st.markdown(message["content"]) # 

if __name__ == "__main__":
    main()