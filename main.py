import streamlit as st
import google.generativeai as genai
import PyPDF2
import tempfile
import os
import uuid
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone, ServerlessSpec
import requests
from bs4 import BeautifulSoup
import re
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "llmforwebscrapyandupload"

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# Streamlit UI Configuration
st.set_page_config(page_title="Ctrl+F on Steroids", layout="wide")
st.title("🔎 RAG IT! ")
st.markdown("Search PDFs & Websites like never before!")

# Sidebar Navigation
option = st.sidebar.radio("Choose a feature:", ["📄 Multi-PDF Q&A", "🌐 Web Scraping & Querying"], index=0)

# Chat Memory Setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

class MultiPDFQuestionAnswerer:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def extract_text_from_pdf(self, uploaded_files):
        for uploaded_file in uploaded_files:
            file_id = str(uuid.uuid4())
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(uploaded_file.getvalue())
                temp_pdf_path = temp_pdf.name
            
            try:
                with open(temp_pdf_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    pdf_text = "".join([page.extract_text() + "\n" for page in pdf_reader.pages])
            finally:
                os.unlink(temp_pdf_path)
            
            self.create_embedding(file_id, pdf_text)

    def create_embedding(self, file_id, pdf_text):
        chunk_size = 500
        chunks = [pdf_text[i:i+chunk_size] for i in range(0, len(pdf_text), chunk_size)]
        embeddings = [self.embedding_model.encode(chunk).tolist() for chunk in chunks]
        vectors = [(f"{file_id}_{i}", embeddings[i], {"text": chunks[i]}) for i in range(len(chunks))]
        index.upsert(vectors)
        st.success(f"Stored {len(chunks)} text chunks in Pinecone.")

    def retrieve_relevant_context(self, query, top_k=3):
        query_embedding = self.embedding_model.encode(query).tolist()
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        return [match.metadata["text"] for match in results.matches]

    def answer_question(self, query):
        contexts = self.retrieve_relevant_context(query)
        if not contexts:
            return "No relevant information found in the PDFs."
        prompt = f"""
        Answer the following question based on the provided context:
        Context:
        {' '.join(contexts)}
        Question: {query}
        Answer:"""
        response = model.generate_content(prompt)
        return response.text if response else "Could not generate an answer."

# Web Scraping Function
def scrape_page(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.extract()
        return re.sub(r'\s+', ' ', soup.get_text()).strip()
    except Exception as e:
        return f"Error scraping {url}: {e}"

# PDF Q&A Section
if option == "📄 Multi-PDF Q&A":
    if "multi_qa_system" not in st.session_state:
        st.session_state.multi_qa_system = MultiPDFQuestionAnswerer()

    uploaded_files = st.file_uploader("📂 Upload PDFs", type=['pdf'], accept_multiple_files=True)
    if uploaded_files:
        with st.spinner("Processing PDFs..."):
            st.session_state.multi_qa_system.extract_text_from_pdf(uploaded_files)

# Web Scraping Section
elif option == "🌐 Web Scraping & Querying":
    website_url = st.text_input("🌍 Enter a website URL:")
    if website_url and st.button("🔍 Scrape Website"):
        with st.spinner("Scraping website..."):
            scraped_content = scrape_page(website_url)
        if scraped_content:
            st.session_state.scraped_data = scraped_content
            st.success("✅ Website scraped successfully!")
        else:
            st.warning("⚠️ No content extracted from the website.")

# Display Chat History
st.markdown("---")
st.subheader("💬 Chat")
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])

# Chat Input Box at the Bottom
query = st.chat_input("Ask a question...")
if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    with st.spinner("Generating answer..."):
        if option == "📄 Multi-PDF Q&A":
            answer = st.session_state.multi_qa_system.answer_question(query)
        elif option == "🌐 Web Scraping & Querying" and "scraped_data" in st.session_state:
            prompt = f"Based on this information: {st.session_state.scraped_data}. Question: {query}"
            response = model.generate_content(prompt)
            answer = response.text if response else "Could not generate an answer."
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)