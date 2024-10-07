# Importing libraries
import os
import time
import uvicorn
import pytesseract
from PyPDF2 import PdfReader
from typing import List, Dict
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.chains import LLMChain
from pdf2image import convert_from_path
from langchain.chains import RetrievalQA
from fastapi import FastAPI, HTTPException
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Define the structure for request body
class QueryRequest(BaseModel):
    query: str

# Define the structure for document details in the response
class DocumentDetails(BaseModel):
    document_name: str
    pages: List[int]

# Define the response structure
class ResponseData(BaseModel):
    answer: str
    relevant_questions: List[str]
    documents: List[DocumentDetails]

class QueryResponse(BaseModel):
    data: ResponseData

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# PDF folder details
pdf_folder = "local_pdfs/"
vectorstore_path = os.path.join(pdf_folder, 'vectorstore.faiss')

# Ensure the folder for storing PDFs exists
os.makedirs(pdf_folder, exist_ok=True)

# Initializing OpenAI Embeddings
embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

# Function to extract text from a PDF file
def extract_text_from_pdf(file_path):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def extract_text_from_image(image):
        return pytesseract.image_to_string(image)

    text_pages = []
    try:
        pdf_reader = PdfReader(file_path)
        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text()
            if page_text.strip():
                text_pages.append({"text": page_text, "page": page_num})
        
        if not text_pages:
            images = convert_from_path(file_path, poppler_path=r"C:\Users\harsh\OneDrive\Desktop\Release-24.07.0-0\poppler-24.07.0\Library\bin")
            for i, image in enumerate(images, 1):
                text_pages.append({"text": extract_text_from_image(image), "page": i})
    except Exception as e:
        print(f"Error extracting text from {file_path}: {str(e)}")
    
    return text_pages

# Load documents and create vectorstore
def create_or_load_vectorstore():
    if not os.listdir(pdf_folder) or not os.path.exists(vectorstore_path):
        print("Creating new vectorstore from PDFs...")
        documents = []
        for filename in os.listdir(pdf_folder):
            if filename.endswith('.pdf'):
                file_path = os.path.join(pdf_folder, filename)
                start_time = time.time()
                text_pages = extract_text_from_pdf(file_path)
                for page in text_pages:
                    documents.append({
                        "page_content": page["text"],
                        "metadata": {
                            "source": file_path,
                            "document_name": filename,
                            "page": page["page"]
                        }
                    })
                time_taken = time.time() - start_time
                print(f"Successfully processed {filename} in {time_taken:.2f} seconds")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
        final_documents = text_splitter.create_documents(
            [doc["page_content"] for doc in documents],
            metadatas=[doc["metadata"] for doc in documents]
        )

        vectorstore = FAISS.from_documents(final_documents, embedding=embeddings)
        vectorstore.save_local(vectorstore_path)
    else:
        print("Loading existing vectorstore...")
        vectorstore = FAISS.load_local(vectorstore_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    
    return vectorstore

try:
    vectorstore = create_or_load_vectorstore()
except Exception as e:
    print('Not able to load Vectorstore:', e)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# Define the custom prompt template
prompt_template = """Answer the user's question based on the given context and chat history, and provide a list of three relevant follow-up questions. Ensure that the answer is concise, clear, and based solely on the provided context and chat history.
You are an AI model trained to answer questions strictly based on the provided context and chat history.
Context: {context}
Chat History: {chat_history}
Answer:
[Provide the answer]
Relevant questions:
[List three relevant questions]
Question: {question}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])

# Initialize ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")

# Set up RetrievalQA chain
retrievalQA = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt, "memory": memory}
)

def decompose_query(query: str) -> List[str]:
    decomposition_template = """
    Given the following question, break it down into smaller, simpler sub-questions that can be answered independently:

    Question: {question}

    Sub-questions:
    1.
    2.
    3.
    """
    decomposition_prompt = PromptTemplate(
        input_variables=["question"],
        template=decomposition_template
    )
    decomposition_chain = LLMChain(llm=llm, prompt=decomposition_prompt)
    
    result = decomposition_chain.run(question=query)
    sub_questions = [q.strip() for q in result.split('\n') if q.strip().startswith(('1.', '2.', '3.'))]
    return sub_questions

# Function to process the query and format the response
def get_answer(query: str) -> Dict:
    sub_queries = decompose_query(query)
    sub_answers = []
    for sub_query in sub_queries:
        sub_result = retrievalQA.invoke({"query": sub_query})
        sub_answers.append(sub_result)
    
    combined_answer = "\n".join([f"Sub-query: {q}\nAnswer: {a['result']}" for q, a in zip(sub_queries, sub_answers)])
    result = retrievalQA.invoke({"query": query})
    split_output = result.get('result', '').split('Answer:')
    answer = split_output[-1].strip() if len(split_output) > 1 else split_output[0].strip()
    response_data = {
        "data": {
            "answer": "",
            "relevant_questions": [],
            "documents": []
        }
    }
    if "The provided context does not contain information about" in answer:
        response_data["data"]["answer"] = "The model does not know the answer to this question."
    else:
        markers = ["Relevant questions:", "Relevant questions related to the user's query:", "relevant questions",
                   "Recommend questions", "recommend questions", "Here are some relevant questions:",
                   "Here are some related questions:", "Other questions you might find useful:", "Other relevant questions:"]
        response_parts = None
        for marker in markers:
            if marker in answer:
                response_parts = answer.split(marker)
                break
        if response_parts:
            ans, reco_ques = response_parts[0].strip(), response_parts[1].strip()
            response_data["data"]["answer"] = ans
            response_data["data"]["relevant_questions"] = [q.strip() for q in reco_ques.split('\n') if q.strip()]
        else:
            response_data["data"]["answer"] = answer
            response_data["data"]["relevant_questions"] = ["No relevant questions provided."]
    
    document_pages = {}
    for doc in result.get("source_documents", []):
        document_name = doc.metadata.get('document_name', 'Unknown')
        page_number = doc.metadata.get('page', 'Unknown')
        if document_name not in document_pages:
            document_pages[document_name] = {"pages": set()}
        if page_number != 'Unknown':
            document_pages[document_name]["pages"].add(page_number)
    
    for document_name, doc_data in document_pages.items():
        response_data["data"]["documents"].append({
            "document_name": document_name,
            "pages": sorted(list(doc_data["pages"]))
        })
    
    return response_data

# FastAPI route to handle the query
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/ask", response_model=QueryResponse)
async def ask(query_request: QueryRequest):
    try:
        result = get_answer(query_request.query)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)