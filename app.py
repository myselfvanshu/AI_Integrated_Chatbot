import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import time

load_dotenv()

# Ensure environment variables are loaded
groq_api_key = os.getenv('GROQ_API_KEY')

# Check if the environment variables are correctly set
if not groq_api_key:
    st.error("GROQ_API_KEY is not set. Please check your .env file.")
    st.stop()

st.title("Chatbot with AI")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt_template = """
Answer the question based on the provided context only.
Please provide a single and accurate response to the question based on the document.
Do not provide multiple answers or numbered lists unless specifically asked for.

Context:
{context}

Question: {input}

Answer:
"""

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        pdf_path = "./Corpus.pdf"
        
        if not os.path.exists(pdf_path):
            st.error(f"PDF file not found at {pdf_path}")
            return

        try:
            st.session_state.loader = PyPDFLoader(pdf_path)
            documents = st.session_state.loader.load()
            
            if len(documents) == 0:
                st.error("No documents were loaded. The PDF might be empty or unreadable.")
                return
            
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(documents)
            
            if len(st.session_state.final_documents) == 0:
                st.error("No text chunks were created. The PDF might not contain extractable text.")
                return
            
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            st.success("Document successfully embedded and ready for questions.")
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

def add_to_history(user_input, bot_response):
    st.session_state.conversation_history.append({"user": user_input, "bot": bot_response})

def get_conversation_context():
    context = ""
    for entry in st.session_state.conversation_history[-5:]:  # Only use last 5 interactions for context
        context += f"User: {entry['user']}\nBot: {entry['bot']}\n"
    return context

def display_conversation_history():
    for entry in st.session_state.conversation_history:
        st.text(f"User: {entry['user']}")
        st.text(f"Bot: {entry['bot']}")
        st.text("--------------------")

prompt1 = st.text_input("Enter Your Question From Document")

if st.button("Document Embedding"):
    vector_embedding()

if prompt1:
    if "vectors" not in st.session_state:
        st.error("Please perform document embedding first.")
    else:
        context = get_conversation_context()
        complete_prompt = ChatPromptTemplate.from_template(prompt_template)
        document_chain = create_stuff_documents_chain(llm, complete_prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1, 'context': context})
        response_time = time.process_time() - start
        
        if 'answer' in response:
            st.write("Answer:")
            st.write(response['answer'])
            add_to_history(prompt1, response['answer'])
        else:
            st.error("No answer was generated. Please try rephrasing your question.")

        st.write("\nConversation History:")
        display_conversation_history()