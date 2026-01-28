# PDF Chat with Citations

Upload research papers, ask questions, and get answers with exact page numbers and quotes from the PDFs.

Built with LangChain, Pinecone, OpenAI, and Streamlit.

## Setup

1. Clone the repo and navigate to the folder
   ```bash
   git clone https://github.com/yourusername/pdf-chat-citations.git
   cd pdf-chat-citations
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Mac/Linux
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys
   ```
   OPENAI_API_KEY=your_key_here
   PINECONE_API_KEY=your_key_here
   ```

## Usage

Run the app:
```bash
streamlit run app.py
```

1. Upload PDF files using the sidebar
2. Click "Process PDFs"
3. Ask questions in the chat

## How It Works

- PDFs are split into chunks and converted to embeddings
- Embeddings are stored in Pinecone vector database
- When you ask a question, relevant chunks are retrieved
- GPT-4 generates an answer with citations to specific pages

## Tech Stack

- **LangChain** - LLM framework
- **Pinecone** - Vector database
- **OpenAI** - Embeddings and GPT-4
- **Streamlit** - Web interface
- **PyPDF** - PDF processing

---

Part of my 30 Days, 30 Projects challenge
