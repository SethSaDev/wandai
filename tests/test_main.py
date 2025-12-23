import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from app.main import app
import json


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_rag_engine():
    with patch('app.main.rag_engine') as mock_engine:
        yield mock_engine


class TestUploadEndpoint:
    def test_upload_document_success(self, client, mock_rag_engine):
        mock_rag_engine.add_document = Mock()
        
        file_content = b"PDF content"
        files = {"file": ("test.pdf", file_content, "application/pdf")}
        
        response = client.post("/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["filename"] == "test.pdf"
    
    def test_upload_document_failure(self, client, mock_rag_engine):
        with patch('app.main.save_uploaded_file') as mock_save:
            mock_save.side_effect = HTTPException(status_code=500, detail="Upload failed")
            
            file_content = b"PDF content"
            files = {"file": ("test.pdf", file_content, "application/pdf")}
            
            response = client.post("/upload", files=files)
            
            assert response.status_code == 500
    
    def test_upload_without_file(self, client):
        response = client.post("/upload")
        
        assert response.status_code == 422


class TestAskEndpoint:
    def test_ask_question_success(self, client, mock_rag_engine):
        mock_response = json.dumps({
            "answer": "AI is artificial intelligence",
            "confidence": 0.9,
            "is_complete": True,
            "missing_info": None,
            "enrichment_suggestions": []
        })
        
        mock_rag_engine.get_answer = AsyncMock(return_value=mock_response)
        
        response = client.post(
            "/ask",
            json={"question": "What is AI?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["confidence"] == 0.9
        assert data["is_complete"] is True
    
    def test_ask_question_no_documents(self, client, mock_rag_engine):
        mock_response = json.dumps({
            "answer": "No documents have been indexed yet.",
            "confidence": 0.0,
            "is_complete": False,
            "missing_info": "No documents in knowledge base.",
            "enrichment_suggestions": ["Upload documents to build the knowledge base"]
        })
        
        mock_rag_engine.get_answer = AsyncMock(return_value=mock_response)
        
        response = client.post(
            "/ask",
            json={"question": "What is AI?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["confidence"] == 0.0
        assert data["is_complete"] is False
    
    def test_ask_question_failure(self, client, mock_rag_engine):
        mock_rag_engine.get_answer = AsyncMock(side_effect=HTTPException(status_code=500, detail="Query failed"))
        
        response = client.post(
            "/ask",
            json={"question": "What is AI?"}
        )
        
        assert response.status_code == 500
    
    def test_ask_without_question(self, client):
        response = client.post("/ask", json={})
        
        assert response.status_code == 422
    
    def test_ask_with_empty_question(self, client, mock_rag_engine):
        mock_response = json.dumps({
            "answer": "Please provide a question",
            "confidence": 0.0,
            "is_complete": False,
            "missing_info": None,
            "enrichment_suggestions": []
        })
        
        mock_rag_engine.get_answer = AsyncMock(return_value=mock_response)
        
        response = client.post(
            "/ask",
            json={"question": ""}
        )
        
        assert response.status_code == 200


class TestMiddleware:
    def test_middleware_logs_slow_requests(self, client, mock_rag_engine):
        mock_response = json.dumps({
            "answer": "test",
            "confidence": 0.5,
            "is_complete": True,
            "missing_info": None,
            "enrichment_suggestions": []
        })
        
        mock_rag_engine.get_answer = AsyncMock(return_value=mock_response)
        
        with patch('app.main.time') as mock_time_module:
            mock_time_module.time.side_effect = [0, 6]
            
            response = client.post(
                "/ask",
                json={"question": "test"}
            )
            
            assert response.status_code == 200
    
    def test_middleware_logs_error_responses(self, client, mock_rag_engine):
        mock_rag_engine.get_answer = AsyncMock(side_effect=HTTPException(status_code=500, detail="Error"))
        
        response = client.post(
            "/ask",
            json={"question": "test"}
        )
        
        assert response.status_code == 500


class TestAPIDocumentation:
    def test_openapi_schema_available(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema
    
    def test_docs_endpoint_available(self, client):
        response = client.get("/docs")
        assert response.status_code == 200
