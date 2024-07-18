import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent

# M.MEETARBHAN
# 7/17/2024 
# MULTI DOCUMENT RETREIVAL AUGMENTED GENERATION SYSTEM FOR BDO GPT 
#

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

st.title("BDO-GPT")
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")


if google_api_key is None: 
    st.warning("API KEY NOT FOUND. PLEASE SET GOOGLE_API_KEY ENVIRONMENT VARIABLE :(")
    st.stop()

#=====================================================================================
# Processing PDFs -- Uses PyPdf2
def pdf_read(pdfDoc):
    text =""
    for pdf in pdfDoc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks
#=====================================================================================

## Embeddings -- Embeds using Google Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# IMPLEMENTATION 1 -- Vector Storarge -- Uses ChromaDB 
vector_index = Chroma.from_texts(texts, embeddings).as_retriever()
# IMPLEMENTATION 2 -- Vector Storage -- Uses FAISS
def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")
