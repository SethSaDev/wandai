import logging
import time
from fastapi import FastAPI, UploadFile, File
from .engine import RAGEngine
from .models import QueryRequest, RAGResponse
from .utils import save_uploaded_file

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Local Insight RAG")
rag_engine = RAGEngine()

@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    if process_time > 5.0 or response.status_code >= 400:
        logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}s")
    
    return response

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        temp_path = save_uploaded_file(file)
        rag_engine.add_document(temp_path)
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        logger.error(f"Upload failed for {file.filename}: {e}", exc_info=True)
        raise

@app.post("/ask", response_model=RAGResponse)
async def ask_question(request: QueryRequest):
    try:
        raw_response = await rag_engine.get_answer(request.question)
        return RAGResponse.model_validate_json(raw_response)
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise