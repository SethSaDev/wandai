import pytest
import json
import os
import tempfile
from io import BytesIO
from unittest.mock import Mock, MagicMock, patch
from app.utils import (
    generate_cache_key,
    extract_json_from_text,
    save_uploaded_file,
    create_no_documents_response,
    create_no_results_response,
    filter_relevant_documents,
    build_rag_prompt
)


class TestGenerateCacheKey:
    def test_cache_key_with_question_only(self):
        key = generate_cache_key("What is AI?")
        assert isinstance(key, str)
        assert len(key) == 32
    
    def test_cache_key_with_context(self):
        key = generate_cache_key("What is AI?", "AI is artificial intelligence")
        assert isinstance(key, str)
        assert len(key) == 32
    
    def test_same_input_same_key(self):
        key1 = generate_cache_key("test question")
        key2 = generate_cache_key("test question")
        assert key1 == key2
    
    def test_different_input_different_key(self):
        key1 = generate_cache_key("question 1")
        key2 = generate_cache_key("question 2")
        assert key1 != key2


class TestExtractJsonFromText:
    def test_extract_valid_json(self):
        text = '{"answer": "test", "confidence": 0.9}'
        result = extract_json_from_text(text)
        assert result == text
    
    def test_extract_json_with_surrounding_text(self):
        text = 'Here is the answer: {"answer": "test", "confidence": 0.9} End of response'
        result = extract_json_from_text(text)
        assert '{"answer": "test", "confidence": 0.9}' in result
    
    def test_extract_json_with_newlines(self):
        text = '''Some text
        {
            "answer": "test",
            "confidence": 0.9
        }
        More text'''
        result = extract_json_from_text(text)
        assert "answer" in result
        assert "confidence" in result
    
    def test_no_json_returns_original(self):
        text = "No JSON here"
        result = extract_json_from_text(text)
        assert result == text


class TestSaveUploadedFile:
    def test_save_uploaded_file(self):
        mock_file = Mock()
        mock_file.filename = "test.pdf"
        mock_file.file = BytesIO(b"test content")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = save_uploaded_file(mock_file, upload_dir=tmpdir)
            assert os.path.exists(result)
            assert result.endswith("test.pdf")
            
            with open(result, 'rb') as f:
                content = f.read()
                assert content == b"test content"
    
    def test_creates_directory_if_not_exists(self):
        mock_file = Mock()
        mock_file.filename = "test.pdf"
        mock_file.file = BytesIO(b"test content")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            upload_dir = os.path.join(tmpdir, "new_uploads")
            result = save_uploaded_file(mock_file, upload_dir=upload_dir)
            assert os.path.exists(upload_dir)
            assert os.path.exists(result)


class TestCreateResponses:
    def test_create_no_documents_response(self):
        response = create_no_documents_response()
        data = json.loads(response)
        assert "answer" in data
        assert data["confidence"] == 0.0
        assert data["is_complete"] is False
        assert "missing_info" in data
        assert isinstance(data["enrichment_suggestions"], list)
    
    def test_create_no_results_response(self):
        response = create_no_results_response()
        data = json.loads(response)
        assert "answer" in data
        assert data["confidence"] == 0.0
        assert data["is_complete"] is False
        assert "missing_info" in data
        assert isinstance(data["enrichment_suggestions"], list)


class TestFilterRelevantDocuments:
    def test_filter_above_threshold(self):
        mock_doc1 = Mock()
        mock_doc2 = Mock()
        mock_doc3 = Mock()
        
        docs_with_scores = [
            (mock_doc1, 0.8),
            (mock_doc2, 0.6),
            (mock_doc3, 0.2)
        ]
        
        result = filter_relevant_documents(docs_with_scores, threshold=0.3)
        assert len(result) == 2
        assert result[0][1] == 0.8
        assert result[1][1] == 0.6
    
    def test_filter_returns_min_docs_when_none_above_threshold(self):
        mock_doc1 = Mock()
        mock_doc2 = Mock()
        
        docs_with_scores = [
            (mock_doc1, 0.1),
            (mock_doc2, 0.05)
        ]
        
        result = filter_relevant_documents(docs_with_scores, threshold=0.3, min_docs=2)
        assert len(result) == 2
    
    def test_filter_respects_max_docs(self):
        docs_with_scores = [(Mock(), 0.9) for _ in range(10)]
        
        result = filter_relevant_documents(docs_with_scores, max_docs=4)
        assert len(result) == 4


class TestBuildRagPrompt:
    def test_build_prompt_with_context_and_question(self):
        context = "AI is artificial intelligence"
        question = "What is AI?"
        
        prompt = build_rag_prompt(context, question)
        assert context in prompt
        assert question in prompt
        assert "CONTEXT:" in prompt
        assert "QUESTION:" in prompt
        assert "JSON format:" in prompt
    
    def test_prompt_includes_json_structure(self):
        prompt = build_rag_prompt("context", "question")
        assert "answer" in prompt
        assert "confidence" in prompt
        assert "is_complete" in prompt
        assert "missing_info" in prompt
        assert "enrichment_suggestions" in prompt
