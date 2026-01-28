import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Configuration
INDEX_NAME = "pdf-chat-index"
EMBEDDING_DIMENSION = 1536  # OpenAI embedding dimension

# Create index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Initialize embeddings and LLM
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Streamlit UI
st.title("📚 PDF Chat with Citations")
st.write("Upload research papers and ask questions with exact citations!")

# Sidebar for PDF upload
with st.sidebar:
    st.header("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True
    )
    
    if st.button("Process PDFs"):
        if uploaded_files:
            with st.spinner("Processing PDFs..."):
                all_documents = []
                
                for uploaded_file in uploaded_files:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    
                    # Load PDF
                    loader = PyPDFLoader(tmp_path)
                    documents = loader.load()
                    
                    # Add source filename to metadata
                    for doc in documents:
                        doc.metadata['source'] = uploaded_file.name
                    
                    all_documents.extend(documents)
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
                
                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                splits = text_splitter.split_documents(all_documents)
                
                # Store in Pinecone
                vectorstore = PineconeVectorStore.from_documents(
                    documents=splits,
                    embedding=embeddings,
                    index_name=INDEX_NAME
                )
                
                st.success(f"Processed {len(uploaded_files)} PDF(s) with {len(splits)} chunks!")
                st.session_state['vectorstore'] = vectorstore

# Main chat interface
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your PDFs"):
    # Check if PDFs have been processed
    if 'vectorstore' not in st.session_state:
        st.warning("Please upload and process PDFs first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get relevant documents
        vectorstore = st.session_state['vectorstore']
        relevant_docs = vectorstore.similarity_search(prompt, k=4)
        
        # Create context with citations
        context_parts = []
        for i, doc in enumerate(relevant_docs):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            context_parts.append(
                f"[Source {i+1}: {source}, Page {page}]\n{doc.page_content}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Create prompt with citations
        system_prompt = f"""You are a helpful research assistant. Answer the question based on the provided context.
        
IMPORTANT: For every piece of information you provide, cite the source using this format:
[Source X, Page Y]

Be specific and quote relevant passages when helpful.

Context:
{context}

Question: {prompt}

Answer with citations:"""
        
        # Get response from LLM
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            response = llm.invoke(system_prompt)
            answer = response.content
            
            # Add citations section
            citations = "\n\n---\n**📎 Sources:**\n"
            for i, doc in enumerate(relevant_docs):
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'Unknown')
                preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                citations += f"\n**[Source {i+1}]** {source} - Page {page}\n> {preview}\n"
            
            full_response = str(answer) + str(citations)
            message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# Add a clear chat button
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()