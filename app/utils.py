import hashlib
import re
import json
import os
import shutil
from typing import Optional, Tuple, List


def generate_cache_key(question: str, context: Optional[str] = None) -> str:
    """Generate MD5 hash for caching purposes."""
    data = f"{question}:{context}" if context else question
    return hashlib.md5(data.encode()).hexdigest()


def extract_json_from_text(text: str) -> str:
    """Extract JSON object from LLM response text."""
    json_match = re.search(r'\{[^}]*"answer"[^}]*"confidence"[^}]*\}', text, re.DOTALL)
    if not json_match:
        json_match = re.search(r'\{.*?\}', text, re.DOTALL)
    
    return json_match.group() if json_match else text


def save_uploaded_file(file, upload_dir: str = "uploads") -> str:
    """Save uploaded file to disk and return the file path."""
    os.makedirs(upload_dir, exist_ok=True)
    temp_path = os.path.join(upload_dir, file.filename)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return temp_path


def create_no_documents_response() -> str:
    """Create JSON response for when no documents are indexed."""
    return json.dumps({
        "answer": "No documents have been indexed yet. Please upload a document first.",
        "confidence": 0.0,
        "is_complete": False,
        "missing_info": "No documents in knowledge base.",
        "enrichment_suggestions": ["Upload documents to build the knowledge base"]
    })


def create_no_results_response() -> str:
    """Create JSON response for when no relevant documents are found."""
    return json.dumps({
        "answer": "I don't have enough relevant information to answer this question.",
        "confidence": 0.0,
        "is_complete": False,
        "missing_info": "No relevant documents found in the knowledge base.",
        "enrichment_suggestions": ["Upload documents related to this topic"]
    })


def filter_relevant_documents(docs_with_scores: List[Tuple], threshold: float = 0.3, min_docs: int = 2, max_docs: int = 4) -> List[Tuple]:
    """Filter documents by relevance score threshold."""
    relevant_docs = [(doc, score) for doc, score in docs_with_scores if score > threshold]
    
    if not relevant_docs:
        return docs_with_scores[:min_docs]
    
    return relevant_docs[:max_docs]


def build_rag_prompt(context: str, question: str) -> str:
    """Build the RAG prompt template."""
    template = """Answer based on context. Return JSON only.

CONTEXT:
{context}

QUESTION:
{question}

JSON format:
{{
    "answer": "string",
    "confidence": 0.0,
    "is_complete": true,
    "missing_info": null,
    "enrichment_suggestions": []
}}"""
    
    return template.format(context=context, question=question)
