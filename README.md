# Local Insight RAG

A production-ready Retrieval-Augmented Generation (RAG) system built with FastAPI, LangChain, and Ollama. Upload PDF documents and query them using natural language with AI-powered responses.

## Features

- **PDF Document Processing**: Upload and index PDF documents automatically
- **Semantic Search**: FAISS vector store with HuggingFace embeddings
- **Local LLM**: Powered by Ollama (llama3.2:3b)
- **Smart Caching**: Response and embedding caching for faster queries
- **Production-Ready**: Clean logging, error handling, and structured responses
- **RESTful API**: FastAPI with automatic OpenAPI documentation

## Architecture

- **FastAPI**: Web framework with async support
- **LangChain**: RAG pipeline orchestration
- **Ollama**: Local LLM inference (llama3.2:3b)
- **FAISS**: Vector similarity search
- **HuggingFace**: Sentence transformers for embeddings (all-MiniLM-L6-v2)
- **PyPDF**: PDF document parsing

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- Ollama model: `llama3.2:3b`

### Install Ollama and Model

```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama service
ollama serve

# Pull the required model
ollama pull llama3.2:3b
```

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd wandai
```

### 2. Create Virtual Environment

```bash
python3 -m venv rag-env
source rag-env/bin/activate  # On Windows: rag-env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Pre-Download Models (Optional but Recommended)

```bash
python download_models.py
```

This downloads the sentence-transformers model (~80MB) to avoid delays on first startup.

## Usage

### Start the Server

```bash
uvicorn app.main:app --reload
```

Server will start at `http://localhost:8000`

### API Documentation

Interactive API docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### 1. Upload Document

Upload a PDF document to the knowledge base.

**Endpoint:** `POST /upload`

**Request:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "status": "success",
  "filename": "document.pdf"
}
```

**Features:**
- Automatically extracts text from PDF
- Splits into chunks (500 chars, 50 char overlap)
- Generates embeddings and stores in FAISS index
- Persists to disk for future sessions

---

### 2. Ask Question

Query the knowledge base with natural language.

**Endpoint:** `POST /ask`

**Request:**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main topic of the document?"
  }'
```

**Response:**
```json
{
  "answer": "The main topic is...",
  "confidence": 0.85,
  "is_complete": true,
  "missing_info": null,
  "enrichment_suggestions": []
}
```

**Response Fields:**
- `answer` (string): AI-generated answer based on document context
- `confidence` (float): Confidence score between 0.0 and 1.0
- `is_complete` (boolean): Whether the answer fully satisfies the query
- `missing_info` (string|null): Details on lacking information
- `enrichment_suggestions` (array): Suggestions to improve knowledge base

**Features:**
- Semantic search with relevance filtering
- Context-aware responses
- Automatic caching (30 min TTL)
- Embedding cache (2 hour TTL)

## Project Structure

```
wandai/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application & endpoints
│   ├── engine.py         # RAG engine implementation
│   ├── models.py         # Pydantic models for request/response
│   └── utils.py          # Utility functions
├── download_models.py    # Pre-download script for embeddings
├── requirements.txt      # Python dependencies
├── vectorstore/          # FAISS index (auto-created)
├── uploads/              # Uploaded PDF files (auto-created)
└── README.md
```

## Configuration

### RAG Engine Settings

Edit `app/engine.py` to customize:

```python
# LLM Configuration
self.llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
    num_ctx=1024,        # Context window
    num_predict=256,     # Max tokens to generate
    num_gpu=1,
    num_thread=8,
    keep_alive="30m"
)

# Embedding Model
model_name="sentence-transformers/all-MiniLM-L6-v2"

# Text Splitting
chunk_size=500
chunk_overlap=50

# Cache Settings
response_cache = TTLCache(maxsize=100, ttl=1800)  # 30 min
embedding_cache = TTLCache(maxsize=200, ttl=7200)  # 2 hours
```

## Performance

### Expected Response Times

- **First query**: 20-25 seconds (with pre-downloaded models)
- **Cached query**: <0.5 seconds
- **Document upload**: 2-5 seconds per PDF

### Optimization Tips

1. **Pre-download models** using `download_models.py`
2. **Use GPU** if available (CUDA support)
3. **Adjust chunk size** based on document type
4. **Monitor slow queries** (logged if >15s)

## Testing

The project includes comprehensive unit tests for all components.

### Run All Tests

```bash
source rag-env/bin/activate
pytest tests/ -v
```

### Run Specific Test Files

```bash
# Test utilities
pytest tests/test_utils.py -v

# Test models
pytest tests/test_models.py -v

# Test RAG engine
pytest tests/test_engine.py -v

# Test API endpoints
pytest tests/test_main.py -v
```

### Run Specific Test Classes or Methods

```bash
# Run a specific test class
pytest tests/test_utils.py::TestGenerateCacheKey -v

# Run a specific test method
pytest tests/test_main.py::TestUploadEndpoint::test_upload_document_success -v
```

### Test Coverage

- **test_utils.py**: Utility functions (cache keys, JSON extraction, file handling, response builders)
- **test_models.py**: Pydantic models validation
- **test_engine.py**: RAG engine initialization, document processing, query handling
- **test_main.py**: API endpoints, middleware, error handling

## Logging

Production-grade logging is configured with:
- **INFO**: Initialization, document indexing, request tracking
- **WARNING**: Slow operations, missing documents
- **ERROR**: Exceptions with full stack traces

Logs include:
- Device selection (CPU/GPU)
- Document indexing status
- Slow LLM inference warnings (>15s)
- Request timing for slow/failed requests (>5s or 4xx/5xx)

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

### Model Not Found

```bash
# Pull the required model
ollama pull llama3.2:3b
```

### Clear Vector Store

```bash
rm -rf vectorstore/
# Re-upload documents
```

### Clear Model Cache

```bash
rm -rf ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2
python download_models.py
```

## Dependencies

Core dependencies (see `requirements.txt`):
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `langchain` - RAG framework
- `langchain-ollama` - Ollama integration
- `langchain-huggingface` - HuggingFace embeddings
- `sentence-transformers` - Embedding models
- `faiss-cpu` - Vector similarity search
- `pypdf` - PDF parsing
- `cachetools` - Response caching
- `torch` - PyTorch for embeddings

## Trade-offs (24-Hour Constraint)

Due to the 24-hour development constraint, several pragmatic trade-offs were made to deliver a functional MVP:

### **1. Storage & Persistence**
**Trade-off:** File-based storage (FAISS index on disk) instead of a proper database
- **Why:** Quick setup, no database configuration required
- **Limitation:** No metadata tracking, document versioning, or user management
- **Production Need:** Migrate to PostgreSQL with pgvector or Pinecone for scalability

### **2. Authentication & Authorization**
**Trade-off:** No user authentication or multi-tenancy support
- **Why:** Focus on core RAG functionality first
- **Limitation:** Single shared knowledge base, no access control
- **Production Need:** Add JWT authentication, user-specific document collections

### **3. Document Processing**
**Trade-off:** PDF-only support with basic text extraction
- **Why:** PyPDF is lightweight and handles most common PDFs
- **Limitation:** No support for images, tables, or complex layouts; no OCR for scanned PDFs
- **Production Need:** Add support for DOCX, images (OCR), structured data extraction

### **4. Caching Strategy**
**Trade-off:** In-memory TTL cache instead of distributed cache
- **Why:** Simple implementation, no external dependencies
- **Limitation:** Cache lost on restart, not shared across instances
- **Production Need:** Redis for distributed caching across multiple instances

### **What Worked Well Despite Constraints:**
- Clean separation of concerns (main, engine, models, utils)
- Production-grade logging from the start
- Utility function extraction for maintainability
- Caching for performance optimization
- Structured JSON responses for API consistency

## Future Recommendations

### Auto-Enrichment System

Implement automatic data enrichment when the system detects missing information:

**Recommended Approach:**
1. **Detection**: Modify `RAGResponse` to include a `needs_enrichment` flag when confidence is low or information is incomplete
2. **External Sources**: Integrate trusted APIs (e.g., Wikipedia API, domain-specific knowledge bases)
3. **Workflow**:
   - Detect missing information in the response
   - Query external source with structured queries
   - Validate and merge external data with existing knowledge base
   - Re-index enriched content automatically

**Implementation Suggestions:**
- Add new endpoint: `POST /enrich` to trigger manual enrichment
- Create `app/enrichment.py` module with:
  - `ExternalSourceConnector` class for API integrations
  - `EnrichmentValidator` to verify data quality
  - `AutoEnrichmentEngine` to orchestrate the process
- Add configuration for trusted sources in `app/config.py`
- Log all enrichment activities for audit trail

**Example Flow:**
```python
# When confidence < 0.5 or missing_info is not None
if response.confidence < 0.5:
    enrichment_suggestions = await enrichment_engine.suggest_sources(query)
    # Optionally auto-fetch from trusted sources
    enriched_data = await enrichment_engine.fetch_and_validate(enrichment_suggestions)
    # Add to knowledge base
    rag_engine.add_enriched_content(enriched_data)
```

### Answer Quality Rating System

Implement user feedback loop to continuously improve the RAG pipeline:

**Recommended Approach:**
1. **Feedback Collection**: Add rating endpoint to capture user satisfaction
2. **Storage**: Store ratings with query-response pairs for analysis
3. **Analytics**: Track metrics to identify weak areas
4. **Improvement**: Use feedback to tune retrieval and generation parameters

**Implementation Suggestions:**
- Add new endpoint: `POST /rate` to accept user ratings
- Create `app/feedback.py` module with:
  - `FeedbackStore` class (SQLite or PostgreSQL)
  - `QualityAnalyzer` to compute metrics
  - `PipelineOptimizer` to suggest parameter adjustments
- Add database schema:
  ```sql
  CREATE TABLE feedback (
    id INTEGER PRIMARY KEY,
    question TEXT,
    answer TEXT,
    rating INTEGER,  -- 1-5 scale
    feedback_text TEXT,
    timestamp DATETIME,
    response_time FLOAT,
    confidence FLOAT
  );
  ```

**Example Endpoint:**
```python
@app.post("/rate")
async def rate_answer(rating: RatingRequest):
    # rating: {question_id, rating (1-5), feedback_text}
    feedback_store.save_rating(rating)
    
    # Analyze patterns
    if rating.rating <= 2:
        # Log low-quality responses for review
        logger.warning(f"Low rating for question: {rating.question_id}")
    
    return {"status": "success", "message": "Thank you for your feedback"}
```

**Metrics to Track:**
- Average rating per document source
- Correlation between confidence scores and user ratings
- Common patterns in low-rated responses
- Response time vs. quality correlation

**Pipeline Improvements Based on Feedback:**
- Adjust `chunk_size` and `chunk_overlap` based on rating patterns
- Fine-tune relevance threshold for document filtering
- Identify documents that consistently produce low-quality answers
- A/B test different prompt templates and track ratings
