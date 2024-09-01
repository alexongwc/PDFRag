import streamlit as st
import time
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from PyPDF2 import PdfReader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader

# Application title and information
st.title("SG Road Traffic Offence Assistant :car: :robot_face:")
with st.expander('About This App', expanded=True):
    st.write('''
             This chatbot provides answers related to road traffic offences in Singapore.
             All information is sourced from the Road Traffic Act (RTA) available at https://sso.agc.gov.sg/Act/RTA1961.
             '''
    )

def setup_components(api_key):
    """
    Setup embeddings, language model, and vector store.
    """
    # Initialize OpenAI embeddings and language model
    embedding_model = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=api_key)
    language_model = ChatOpenAI(model="gpt-3.5-turbo-0125", openai_api_key=api_key)

    # Extract text from the provided PDF document
    pdf_file_path = "./data/rta.pdf"
    with open(pdf_file_path, "rb") as file:
        reader = PdfReader(file)
        pdf_text = ""
        for page in reader.pages:
            page_content = page.extract_text()
            if page_content:
                pdf_text += page_content + "\n\n"

    # Prepare DataFrame for document processing
    document_data = [{"content": pdf_text}]
    document_df = pd.DataFrame(document_data)

    # Load documents and split them into chunks
    document_loader = DataFrameLoader(document_df, page_content_column="content")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    processed_docs = document_loader.load_and_split(text_splitter=splitter)

    # Create a vector store with the processed documents
    vector_store = Chroma.from_documents(documents=processed_docs, embedding=embedding_model, persist_directory='db')

    return vector_store, language_model

def stream_output_text(text):
    for word in text.split():
        yield word + " "
        time.sleep(0.05)

def format_document_content(documents):
    return "\n\n".join(doc.page_content for doc in documents)

def extract_metadata(documents):
    return [doc.metadata for doc in documents]

def generate_response(query, rag_chain):
    """
    Generate a response based on the user query and the RAG chain.
    """
    return rag_chain.invoke(query)

def get_answer_chain(llm, vector_store):
    prompt_template = """You are a knowledgeable assistant for answering questions.
    Given the following context, respond to the user's question accurately.
    Context: {context}

    Answer the following query using only the provided context. Do not make up information.
    If unsure, state that you do not know.

    Include the source URL from the metadata: {metadata}
    
    Query: {query}

    Response:"""
    
    retriever = vector_store.as_retriever()
    
    prompt = PromptTemplate(
        input_variables=["context", "metadata", "query"],
        template=prompt_template)

    answer_chain = (
        {"context": retriever | format_document_content,
        "metadata": retriever | extract_metadata,
        "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return answer_chain

def display_chat_interface(answer_chain):
    # Initialize chat session
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = [{"role": "assistant",
                                                  "content": "Welcome! How can I assist you with road traffic offences?"}]
    
    # Display past conversation
    for chat_message in st.session_state.conversation_history:
        with st.chat_message(chat_message["role"]):
            st.markdown(chat_message["content"])

    # Handle new user input
    if user_input := st.chat_input():
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        
        # Generate and display bot's response
        bot_response = generate_response(user_input, answer_chain)
        with st.chat_message("assistant"):
            response_text = st.write_stream(stream_output_text(bot_response))
        
        st.session_state.conversation_history.append({"role": "assistant", "content": response_text})

def main():
    ready = True
    
    with st.sidebar:
        api_key = st.text_input("Enter your OpenAI API Key:", key="api_key_input", type="password")
        
    if not api_key:
        st.info("Please enter your OpenAI API key to proceed.")
        ready = False
        
    if ready:
        vector_store, language_model = setup_components(api_key)
        with st.sidebar:
            st.success("API Authentication Successful!")
        answer_chain = get_answer_chain(llm=language_model, vector_store=vector_store)
        display_chat_interface(answer_chain)
        
    else:
        st.stop()
        
if __name__ == "__main__":
    main()