import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from app.engine import RAGEngine
import json


class TestRAGEngineInit:
    @patch('app.engine.ChatOllama')
    @patch('app.engine.HuggingFaceEmbeddings')
    @patch('app.engine.torch')
    @patch('app.engine.os.path.exists')
    def test_engine_initialization_no_existing_index(self, mock_exists, mock_torch, mock_embeddings, mock_llm):
        mock_exists.return_value = False
        mock_torch.cuda.is_available.return_value = False
        
        engine = RAGEngine()
        
        assert engine.db is None
        assert engine.response_cache is not None
        assert engine.embedding_cache is not None
        mock_llm.assert_called_once()
        mock_embeddings.assert_called_once()
    
    @patch('app.engine.ChatOllama')
    @patch('app.engine.HuggingFaceEmbeddings')
    @patch('app.engine.torch')
    @patch('app.engine.os.path.exists')
    @patch('app.engine.FAISS.load_local')
    def test_engine_initialization_with_existing_index(self, mock_load, mock_exists, mock_torch, mock_embeddings, mock_llm):
        mock_exists.return_value = True
        mock_torch.cuda.is_available.return_value = False
        
        mock_db = Mock()
        mock_db.index.ntotal = 100
        mock_load.return_value = mock_db
        
        engine = RAGEngine()
        
        assert engine.db is not None
        mock_load.assert_called_once()
    
    @patch('app.engine.ChatOllama')
    @patch('app.engine.HuggingFaceEmbeddings')
    @patch('app.engine.torch')
    @patch('app.engine.os.path.exists')
    def test_engine_detects_cuda(self, mock_exists, mock_torch, mock_embeddings, mock_llm):
        mock_exists.return_value = False
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GPU"
        
        engine = RAGEngine()
        
        mock_torch.cuda.is_available.assert_called()


class TestRAGEngineAddDocument:
    @patch('app.engine.ChatOllama')
    @patch('app.engine.HuggingFaceEmbeddings')
    @patch('app.engine.torch')
    @patch('app.engine.os.path.exists')
    @patch('app.engine.PyPDFLoader')
    @patch('app.engine.FAISS.from_documents')
    @patch('app.engine.os.makedirs')
    def test_add_document_creates_new_db(self, mock_makedirs, mock_faiss, mock_loader, mock_exists, mock_torch, mock_embeddings, mock_llm):
        mock_exists.return_value = False
        mock_torch.cuda.is_available.return_value = False
        
        mock_page = Mock()
        mock_page.page_content = "Test content"
        mock_page.metadata = {"source": "test.pdf", "page": 0}
        mock_loader.return_value.load.return_value = [mock_page]
        
        mock_db = Mock()
        mock_faiss.return_value = mock_db
        
        engine = RAGEngine()
        engine.add_document("test.pdf")
        
        assert engine.db is not None
        mock_loader.assert_called_once_with("test.pdf")
        mock_db.save_local.assert_called_once()
    
    @patch('app.engine.ChatOllama')
    @patch('app.engine.HuggingFaceEmbeddings')
    @patch('app.engine.torch')
    @patch('app.engine.os.path.exists')
    @patch('app.engine.PyPDFLoader')
    def test_add_document_handles_error(self, mock_loader, mock_exists, mock_torch, mock_embeddings, mock_llm):
        mock_exists.return_value = False
        mock_torch.cuda.is_available.return_value = False
        
        mock_loader.side_effect = Exception("PDF load error")
        
        engine = RAGEngine()
        
        with pytest.raises(Exception):
            engine.add_document("bad.pdf")


class TestRAGEngineGetAnswer:
    @patch('app.engine.ChatOllama')
    @patch('app.engine.HuggingFaceEmbeddings')
    @patch('app.engine.torch')
    @patch('app.engine.os.path.exists')
    @pytest.mark.asyncio
    async def test_get_answer_no_db(self, mock_exists, mock_torch, mock_embeddings, mock_llm):
        mock_exists.return_value = False
        mock_torch.cuda.is_available.return_value = False
        
        engine = RAGEngine()
        response = await engine.get_answer("What is AI?")
        
        data = json.loads(response)
        assert data["confidence"] == 0.0
        assert "No documents" in data["answer"]
    
    @patch('app.engine.ChatOllama')
    @patch('app.engine.HuggingFaceEmbeddings')
    @patch('app.engine.torch')
    @patch('app.engine.os.path.exists')
    @pytest.mark.asyncio
    async def test_get_answer_with_cache_hit(self, mock_exists, mock_torch, mock_embeddings, mock_llm):
        mock_exists.return_value = False
        mock_torch.cuda.is_available.return_value = False
        
        engine = RAGEngine()
        engine.db = Mock()
        
        cached_response = '{"answer": "cached", "confidence": 0.9, "is_complete": true, "missing_info": null, "enrichment_suggestions": []}'
        
        from app.utils import generate_cache_key
        cache_key = generate_cache_key("test question")
        engine.response_cache[cache_key] = cached_response
        
        response = await engine.get_answer("test question")
        assert response == cached_response
    
    @patch('app.engine.ChatOllama')
    @patch('app.engine.HuggingFaceEmbeddings')
    @patch('app.engine.torch')
    @patch('app.engine.os.path.exists')
    @pytest.mark.asyncio
    async def test_get_answer_no_documents_found(self, mock_exists, mock_torch, mock_embeddings, mock_llm):
        mock_exists.return_value = False
        mock_torch.cuda.is_available.return_value = False
        
        engine = RAGEngine()
        mock_db = Mock()
        mock_db.similarity_search_with_score_by_vector.return_value = []
        engine.db = mock_db
        
        mock_embeddings_instance = Mock()
        mock_embeddings_instance.embed_query.return_value = [0.1] * 384
        engine.embeddings = mock_embeddings_instance
        
        response = await engine.get_answer("test question")
        data = json.loads(response)
        assert data["confidence"] == 0.0
    
    @patch('app.engine.ChatOllama')
    @patch('app.engine.HuggingFaceEmbeddings')
    @patch('app.engine.torch')
    @patch('app.engine.os.path.exists')
    @pytest.mark.asyncio
    async def test_get_answer_success(self, mock_exists, mock_torch, mock_embeddings, mock_llm):
        mock_exists.return_value = False
        mock_torch.cuda.is_available.return_value = False
        
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.content = '{"answer": "AI is artificial intelligence", "confidence": 0.9, "is_complete": true, "missing_info": null, "enrichment_suggestions": []}'
        mock_llm_instance.ainvoke = AsyncMock(return_value=mock_response)
        mock_llm.return_value = mock_llm_instance
        
        engine = RAGEngine()
        
        mock_doc = Mock()
        mock_doc.page_content = "AI is artificial intelligence"
        mock_db = Mock()
        mock_db.similarity_search_with_score_by_vector.return_value = [(mock_doc, 0.8)]
        engine.db = mock_db
        
        mock_embeddings_instance = Mock()
        mock_embeddings_instance.embed_query.return_value = [0.1] * 384
        engine.embeddings = mock_embeddings_instance
        
        response = await engine.get_answer("What is AI?")
        data = json.loads(response)
        assert "answer" in data
        assert "confidence" in data
