import pytest
from pydantic import ValidationError
from app.models import QueryRequest, RAGResponse


class TestQueryRequest:
    def test_valid_query_request(self):
        request = QueryRequest(question="What is AI?")
        assert request.question == "What is AI?"
    
    def test_empty_question_is_valid(self):
        request = QueryRequest(question="")
        assert request.question == ""
    
    def test_missing_question_raises_error(self):
        with pytest.raises(ValidationError):
            QueryRequest()


class TestRAGResponse:
    def test_valid_rag_response(self):
        response = RAGResponse(
            answer="AI is artificial intelligence",
            confidence=0.85,
            is_complete=True,
            missing_info=None,
            enrichment_suggestions=[]
        )
        assert response.answer == "AI is artificial intelligence"
        assert response.confidence == 0.85
        assert response.is_complete is True
        assert response.missing_info is None
        assert response.enrichment_suggestions == []
    
    def test_confidence_must_be_between_0_and_1(self):
        with pytest.raises(ValidationError):
            RAGResponse(
                answer="test",
                confidence=1.5,
                is_complete=True
            )
        
        with pytest.raises(ValidationError):
            RAGResponse(
                answer="test",
                confidence=-0.1,
                is_complete=True
            )
    
    def test_confidence_boundaries(self):
        response_min = RAGResponse(
            answer="test",
            confidence=0.0,
            is_complete=False
        )
        assert response_min.confidence == 0.0
        
        response_max = RAGResponse(
            answer="test",
            confidence=1.0,
            is_complete=True
        )
        assert response_max.confidence == 1.0
    
    def test_optional_fields_have_defaults(self):
        response = RAGResponse(
            answer="test",
            confidence=0.5,
            is_complete=True
        )
        assert response.missing_info is None
        assert response.enrichment_suggestions == []
    
    def test_missing_info_can_be_string(self):
        response = RAGResponse(
            answer="test",
            confidence=0.5,
            is_complete=False,
            missing_info="Need more context"
        )
        assert response.missing_info == "Need more context"
    
    def test_enrichment_suggestions_is_list(self):
        response = RAGResponse(
            answer="test",
            confidence=0.5,
            is_complete=False,
            enrichment_suggestions=["Add more documents", "Upload PDFs"]
        )
        assert len(response.enrichment_suggestions) == 2
        assert "Add more documents" in response.enrichment_suggestions
    
    def test_model_validate_json(self):
        json_str = '{"answer": "test", "confidence": 0.9, "is_complete": true, "missing_info": null, "enrichment_suggestions": []}'
        response = RAGResponse.model_validate_json(json_str)
        assert response.answer == "test"
        assert response.confidence == 0.9
        assert response.is_complete is True
