# PDF Multilingual Retrieval-Augmented Generation (RAG) System

This project implements a Retrieval-Augmented Generation (RAG) system that processes multilingual PDFs, extracts relevant information using OCR, and answers queries through integration with OpenAI's LLMs.

## Features

- **PDF Parsing**: Extracts text and image-based content using `PyPDF2` and `pytesseract`.
- **OCR Support**: Uses `pytesseract` to handle scanned PDF files.
- **Multilingual Support**: Handles multilingual documents and queries.
- **LLM Integration**: OpenAI GPT-4 model integration for answering queries.
- **Vector Search**: Implements FAISS-based vector search for fast retrieval of relevant documents.
- **Query Decomposition**: Breaks down complex queries into smaller sub-questions.
- **FastAPI**: Provides a simple API to interact with the system.

## Tech Stack

- **Backend**: FastAPI
- **LLM Framework**: LangChain, OpenAI's GPT-4
- **Embeddings**: FAISS for vector search
- **OCR**: `pytesseract`
- **PDF Processing**: `PyPDF2`, `pdf2image`
- **Environment**: `.env` for managing API keys

## Prerequisites

- Python 3.8+
- Install dependencies from the `requirements.txt`
- Install Tesseract OCR
- Install Poppler

### Tesseract Installation

1. Download and install Tesseract OCR from [here](https://github.com/tesseract-ocr/tesseract).
2. Set the `pytesseract` path in your environment or code (as shown in the `extract_text_from_pdf` function).

### Poppler Installation

1. Download and install Poppler binaries from [here](https://github.com/oschwartz10612/poppler-windows).
2. Ensure the path to Poppler is provided in the code when converting PDFs to images.

### Setup Instructions

1. Clone the repository

git clone https://github.com/yourusername/pdf-rag-multilingual-system.git
cd pdf-rag-multilingual-system

2. Create a Virtual Environment
   
python -m venv venv
venv\Scripts\activate

3. Install the dependencies

pip install -r requirements.txt

4. Set up environment variables

Create a .env file in the project root with the following variables:

OPENAI_API_KEY=your_openai_api_key

5. Run the API

python app.py
The API should now be running at http://localhost:8000.

6. API Endpoints

Health Check

GET /health

Description: Check if the API is running.

Response:

{
  "status": "healthy"
}

Ask a Query

POST /ask

### Documentation 

https://harshitraizada.atlassian.net/wiki/external/YWE5YmE0Zjk4ZWM4NDY5ZjkyYjgyMDAwMjUxZWE5ZDI
