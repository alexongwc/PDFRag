import streamlit as st
from PyPDF2 import PdfReader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import openai

def initialize_setup(openai_api_key: str):
    """
    Initialize the embeddings and vector store using the provided OpenAI API key.
    Also, load and process the PDF document from the specified path.

    Parameters:
    openai_api_key (str): The OpenAI API key entered by the user.

    Returns:
    vector_store: The initialized Chroma vector store with the PDF content embedded, or None if initialization fails.
    """
    try:
        # Initialize OpenAI Embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Initialize Chroma vector store
        vector_store = Chroma(embedding_function=embeddings, persist_directory="chroma_storage")
        
        # Load and process the PDF document
        pdf_path = "./data/rta.pdf"
        with open(pdf_path, "rb") as file:
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        
        # Embed the PDF content into the vector store
        vector_store.add_texts([text], metadatas=[{"filename": "rta.pdf"}])
        vector_store.persist()
        
        st.success("API key validated and document processed successfully!")
        return vector_store
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}. Please check your API key and try again.")
        return None

def retrieve(vector_store, user_prompt):
    """
    Retrieve relevant content based on the user's query.

    Parameters:
    vector_store: The initialized Chroma vector store.
    user_prompt: The user's query.

    Returns:
    retrieved_results: The retrieved content from the document relevant to the query.
    """
    try:
        results = vector_store.search(user_prompt, search_type="similarity")
        
        # Return the most relevant content from the document
        return results
    except Exception as e:
        st.error(f"An error occurred while retrieving content: {str(e)}")
        return []

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("SGroad Chatbot :robot_face:")

    with st.expander("Information", expanded=True):
        st.write(
            "This chatbot focuses on answering questions based on the Singapore Road Traffic Act."
        )

    # Prompt for API Key
    openai_api_key = st.text_input("Enter OpenAI API key", type="password")
    
    if openai_api_key:
        vector_store = initialize_setup(openai_api_key)

        if vector_store:
            # Display the chat input box to ask questions
            user_prompt = st.text_input("Ask a question based on the PDF content:", key="chat_input")

            if user_prompt:
                retrieved_results = retrieve(vector_store, user_prompt)
                if retrieved_results:
                    st.write("Here is what I found:")
                    for result in retrieved_results:
                        st.write(f"Document: {result.metadata['filename']}")
                        st.write(f"Snippet: {result.page_content[:200]}...")

if __name__ == "__main__":
    main()