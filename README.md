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

## Project Structure

```bash
pdf-rag-multilingual-system/
├── app.py                      # Main FastAPI application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .env                        # Environment variables for API keys
├── local_pdfs/                 # Folder to store PDFs
└── vectorstore.faiss           # FAISS index for vector search

