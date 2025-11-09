import streamlit as st
from google import genai
import faiss
import numpy as np
import PyPDF2
import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Tuple
import io
from urllib.parse import urlparse
import time

# Initialize Gemini client
API_KEY = "AIzaSyDGc1NP4O4_0vmTv2jANbcV8EZ1GC1Tyjg"
client = genai.Client(api_key=API_KEY)

# Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_DIM = None  # Will be set dynamically based on actual embeddings

class DocumentProcessor:
    """Process PDFs and URLs to extract text"""
    
    @staticmethod
    def extract_pdf_text(pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
    
    @staticmethod
    def extract_url_text(url: str) -> str:
        """Extract text from URL (supports arXiv and general web pages)"""
        try:
            # Handle arXiv URLs specially
            if "arxiv.org" in url:
                # Try to get PDF version
                if "/abs/" in url:
                    pdf_url = url.replace("/abs/", "/pdf/") + ".pdf"
                else:
                    pdf_url = url
                
                response = requests.get(pdf_url, timeout=30)
                if response.status_code == 200:
                    pdf_file = io.BytesIO(response.content)
                    return DocumentProcessor.extract_pdf_text(pdf_file)
            
            # For other URLs, extract HTML text
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            st.error(f"Error extracting text from URL {url}: {e}")
            return ""

class TextChunker:
    """Split text into chunks for embedding"""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks

class EmbeddingGenerator:
    """Generate embeddings using Gemini"""
    
    def __init__(self, client):
        self.client = client
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 10) -> np.ndarray:
        """Generate embeddings in batches"""
        all_embeddings = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            status_text.text(f"Generating embeddings: {i}/{len(texts)}")
            
            try:
                result = self.client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=batch
                )
                
                for embedding in result.embeddings:
                    all_embeddings.append(embedding.values)
                
                progress_bar.progress(min((i + batch_size) / len(texts), 1.0))
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                st.error(f"Error generating embeddings for batch {i}: {e}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        return np.array(all_embeddings, dtype='float32')

class FAISSVectorStore:
    """FAISS vector store for similarity search"""
    
    def __init__(self, dimension: int = None):
        self.dimension = dimension
        self.index = None
        self.texts = []
        self.metadata = []
    
    def add_documents(self, embeddings: np.ndarray, texts: List[str], metadata: List[Dict]):
        """Add documents to the vector store"""
        # Initialize index with actual embedding dimension
        if self.index is None:
            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)
        
        self.index.add(embeddings)
        self.texts.extend(texts)
        self.metadata.extend(metadata)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, Dict, float]]:
        """Search for similar documents"""
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.texts):
                results.append((self.texts[idx], self.metadata[idx], float(distance)))
        
        return results

class ReportGenerator:
    """Generate analysis reports using Gemini LLM"""
    
    def __init__(self, client):
        self.client = client
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using Gemini"""
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt
            )
            return response.text
        except Exception as e:
            st.error(f"Error generating response: {e}")
            return ""
    
    def generate_summary_report(self, context: str) -> str:
        """Generate summary of papers"""
        prompt = f"""Based on the following research papers context, provide a comprehensive summary report:

Context:
{context[:8000]}

Please provide:
1. Main topics covered in these papers
2. Key authors and their contributions
3. Primary research questions addressed
4. Main findings and conclusions

Format your response in a clear, structured manner."""
        
        return self.generate_response(prompt)
    
    def generate_architecture_report(self, context: str) -> str:
        """Generate report on architectures used"""
        prompt = f"""Based on the following research papers context, analyze the architectures and methodologies:

Context:
{context[:8000]}

Please provide:
1. Technical architectures described (models, frameworks, systems)
2. Methodologies employed
3. Algorithms and techniques used
4. Implementation details if mentioned
5. Performance metrics and results

Format your response with clear sections and technical details."""
        
        return self.generate_response(prompt)
    
    def generate_gaps_report(self, context: str) -> str:
        """Generate report on research gaps"""
        prompt = f"""Based on the following research papers context, identify research gaps and future directions:

Context:
{context[:8000]}

Please provide:
1. Identified limitations in current research
2. Research gaps and unexplored areas
3. Future research directions suggested by authors
4. Potential improvements and open problems
5. Cross-paper insights on what's missing in the field

Include references to specific papers where mentioned. Format your response clearly."""
        
        return self.generate_response(prompt)

# Streamlit UI
def main():
    st.set_page_config(page_title="Research Paper Analyzer", layout="wide", page_icon="📚")
    
    st.title("📚 Research Paper Analysis System")
    st.markdown("*Powered by Gemini AI and FAISS Vector Store*")
    
    # Initialize session state
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'embedding_generator' not in st.session_state:
        st.session_state.embedding_generator = EmbeddingGenerator(client)
    if 'report_generator' not in st.session_state:
        st.session_state.report_generator = ReportGenerator(client)
    if 'all_texts' not in st.session_state:
        st.session_state.all_texts = []
    
    # Sidebar for input
    with st.sidebar:
        st.header("📥 Input Sources")
        st.markdown("*Upload up to 5 documents (PDFs + URLs combined)*")
        
        # PDF uploads
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload research papers in PDF format"
        )
        
        # URL inputs
        st.subheader("Or provide URLs")
        urls = []
        for i in range(5):
            url = st.text_input(f"URL {i+1}", key=f"url_{i}", placeholder="https://arxiv.org/abs/...")
            if url:
                urls.append(url)
        
        process_button = st.button("🔄 Process Documents", type="primary")
    
    # Main content area
    if process_button:
        total_inputs = len(uploaded_files) + len(urls)
        
        if total_inputs == 0:
            st.error("Please provide at least one PDF or URL!")
            return
        
        if total_inputs > 5:
            st.error("Maximum 5 inputs allowed (PDFs + URLs combined)!")
            return
        
        with st.spinner("Processing documents..."):
            # Process documents
            all_texts = []
            all_chunks = []
            all_metadata = []
            
            doc_processor = DocumentProcessor()
            text_chunker = TextChunker()
            
            # Process PDFs
            for idx, pdf_file in enumerate(uploaded_files):
                st.info(f"Processing PDF: {pdf_file.name}")
                text = doc_processor.extract_pdf_text(pdf_file)
                if text:
                    chunks = text_chunker.chunk_text(text)
                    all_texts.append(text)
                    all_chunks.extend(chunks)
                    all_metadata.extend([{
                        'source': pdf_file.name,
                        'type': 'pdf',
                        'chunk_id': i
                    } for i in range(len(chunks))])
            
            # Process URLs
            for idx, url in enumerate(urls):
                st.info(f"Processing URL: {url}")
                text = doc_processor.extract_url_text(url)
                if text:
                    chunks = text_chunker.chunk_text(text)
                    all_texts.append(text)
                    all_chunks.extend(chunks)
                    all_metadata.extend([{
                        'source': url,
                        'type': 'url',
                        'chunk_id': i
                    } for i in range(len(chunks))])
            
            if not all_chunks:
                st.error("No text extracted from documents!")
                return
            
            st.success(f"Extracted {len(all_chunks)} chunks from {total_inputs} documents")
            
            # Generate embeddings
            st.info("Generating embeddings...")
            embeddings = st.session_state.embedding_generator.generate_embeddings(all_chunks)
            
            # Create vector store
            st.info("Building vector store...")
            vector_store = FAISSVectorStore()
            vector_store.add_documents(embeddings, all_chunks, all_metadata)
            
            st.session_state.vector_store = vector_store
            st.session_state.all_texts = all_texts
            
            st.success("✅ Documents processed successfully!")
    
    # Display tabs for different functionalities
    if st.session_state.vector_store is not None:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💬 Q&A", "📊 Summary Report", "🏗️ Architecture Report", "🔍 Research Gaps", "📈 Statistics"
        ])
        
        with tab1:
            st.header("Ask Questions")
            question = st.text_input("Enter your question:", placeholder="What are the main findings?")
            
            if st.button("Get Answer", key="qa_button"):
                if question:
                    with st.spinner("Searching for answer..."):
                        # Generate question embedding
                        q_result = client.models.embed_content(
                            model="gemini-embedding-001",
                            contents=[question]
                        )
                        q_embedding = np.array([q_result.embeddings[0].values], dtype='float32')
                        
                        # Search vector store
                        results = st.session_state.vector_store.search(q_embedding, k=5)
                        
                        # Build context
                        context = "\n\n".join([f"Source: {meta['source']}\n{text}" 
                                              for text, meta, _ in results])
                        
                        # Generate answer
                        prompt = f"""Based on the following context from research papers, answer the question.

Context:
{context}

Question: {question}

Please provide a detailed answer based on the context provided. Include source references."""
                        
                        answer = st.session_state.report_generator.generate_response(prompt)
                        
                        st.markdown("### Answer")
                        st.markdown(answer)
                        
                        with st.expander("View Source Chunks"):
                            for i, (text, meta, distance) in enumerate(results):
                                st.markdown(f"**Source {i+1}:** {meta['source']}")
                                st.markdown(f"*Distance: {distance:.4f}*")
                                st.text(text[:300] + "...")
                                st.divider()
        
        with tab2:
            st.header("📊 Summary Report")
            if st.button("Generate Summary Report", key="summary_button"):
                with st.spinner("Generating summary report..."):
                    context = "\n\n".join(st.session_state.all_texts)
                    report = st.session_state.report_generator.generate_summary_report(context)
                    st.markdown(report)
                    
                    # Download button
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name="summary_report.txt",
                        mime="text/plain"
                    )
        
        with tab3:
            st.header("🏗️ Architecture & Methodology Report")
            if st.button("Generate Architecture Report", key="arch_button"):
                with st.spinner("Generating architecture report..."):
                    context = "\n\n".join(st.session_state.all_texts)
                    report = st.session_state.report_generator.generate_architecture_report(context)
                    st.markdown(report)
                    
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name="architecture_report.txt",
                        mime="text/plain"
                    )
        
        with tab4:
            st.header("🔍 Research Gaps & Future Directions")
            if st.button("Generate Gaps Report", key="gaps_button"):
                with st.spinner("Generating research gaps report..."):
                    context = "\n\n".join(st.session_state.all_texts)
                    report = st.session_state.report_generator.generate_gaps_report(context)
                    st.markdown(report)
                    
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name="research_gaps_report.txt",
                        mime="text/plain"
                    )
        
        with tab5:
            st.header("📈 System Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Documents Processed", len(st.session_state.all_texts))
            
            with col2:
                st.metric("Total Chunks", st.session_state.vector_store.index.ntotal)
            
            with col3:
                st.metric("Embedding Dimension", st.session_state.vector_store.dimension)
            
            st.subheader("Document Sources")
            sources = [meta['source'] for meta in st.session_state.vector_store.metadata]
            unique_sources = list(set(sources))
            for source in unique_sources:
                count = sources.count(source)
                st.write(f"- **{source}**: {count} chunks")
    
    else:
        st.info("👈 Please upload documents or provide URLs from the sidebar to get started!")
        
        st.markdown("""
        ### Features:
        - 📄 Upload up to 5 PDFs and/or URLs (combined)
        - 🔗 Support for arXiv papers and web pages
        - 🤖 Gemini embeddings for semantic search
        - 💾 FAISS vector store for efficient retrieval
        - 💬 Ask questions about your documents
        - 📊 Generate comprehensive analysis reports:
          - Summary of papers and key findings
          - Architecture and methodology analysis
          - Research gaps and future directions
        - 📥 Download reports for offline use
        """)

if __name__ == "__main__":
    main()