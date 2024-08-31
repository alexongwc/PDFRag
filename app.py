import streamlit as st
from PyPDF2 import PdfReader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from fpdf import FPDF
import openai

def initialize_setup(openai_api_key: str):
    """
    Initialize the embeddings and vector store using the provided OpenAI API key.

    Parameters:
    openai_api_key (str): The OpenAI API key entered by the user.

    Returns:
    vector_store: The initialized Chroma vector store, or None if initialization fails.
    """
    try:
        # Initialize OpenAI Embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Initialize Chroma vector store
        vector_store = Chroma(
            embedding_function=embeddings, persist_directory="chroma_storage"
        )
        
        st.sidebar.success("API key validated successfully!")
        return vector_store
    
    except Exception as e:
        st.sidebar.error(
            f"An error occurred: {str(e)}. Please check your API key and try again."
        )
        return None

def upload_and_embed_files(vector_store):
    """
    Handle PDF file uploads, extract text, and embed it into the vector store.

    Parameters:
    vector_store: The initialized Chroma vector store.

    Returns:
    document_metadata: A list of metadata for the uploaded documents.
    """
    document_metadata = []
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF files", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            st.write(f"Uploaded file: {file.name}")

            # Read PDF file and extract text
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # Add the text to the Chroma vector store
            vector_store.add_texts([text], metadatas=[{"filename": file.name}])
            
            # Collect metadata
            document_metadata.append({"filename": file.name, "content": text})

            st.write(f"Processed and stored: {file.name}")

        # Ensure data is saved
        vector_store.persist()

    return document_metadata

def rank_documents(vector_store, user_prompt):
    """
    Rank the documents based on their relevance to the user's query.

    Parameters:
    vector_store: The initialized Chroma vector store.
    user_prompt: The user's query.

    Returns:
    ranked_results: A list of documents ranked by relevance.
    """
    try:
        results = vector_store.search(user_prompt)
        ranked_results = sorted(results, key=lambda x: x["score"], reverse=True)
        return ranked_results
    except Exception as e:
        st.error(f"An error occurred while ranking documents: {str(e)}")
        return []

def generate_relevance_report(ranked_results):
    """
    Generate a PDF report with the ranking and reasons for relevance.

    Parameters:
    ranked_results: A list of documents ranked by relevance.

    Returns:
    None
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Document Relevance Report", ln=True, align='C')

    for idx, result in enumerate(ranked_results, start=1):
        pdf.cell(200, 10, txt=f"Rank {idx}: {result['metadata']['filename']}", ln=True)
        pdf.multi_cell(0, 10, txt=f"Reason for relevance: {result['content'][:200]}...")  # Add a snippet of the content

    # Save the PDF
    pdf.output("relevance_report.pdf")

    st.success("Relevance report generated: relevance_report.pdf")

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("PDF Chatbot :robot_face:")

    with st.expander("Information", expanded=True):
        st.write(
            "This chatbot focuses on answering questions based on the content of PDF files."
        )

    st.sidebar.title("PDF Upload for RAG Bot Training")

    # Step 1: Prompt for API Key
    openai_api_key = st.sidebar.text_input("Enter OpenAI API key", type="password")

    # Show the file uploader regardless, but only process if API key is provided
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF files", type="pdf", accept_multiple_files=True
    )

    if openai_api_key:
        vector_store = initialize_setup(openai_api_key)
        
        if vector_store and uploaded_files:
            document_metadata = upload_and_embed_files(vector_store)

            if document_metadata:
                user_prompt = st.text_input("Ask a question based on the PDF content:")

                if user_prompt:
                    ranked_results = rank_documents(vector_store, user_prompt)
                    if ranked_results:
                        generate_relevance_report(ranked_results)
        else:
            st.sidebar.warning("Please upload PDF files to proceed.")

    else:
        st.sidebar.info("Please enter your OpenAI API key to continue.")

if __name__ == "__main__":
    main()