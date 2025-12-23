"""
Pre-download sentence-transformers model to avoid first-run delay.
Run this once before starting the server.
"""

from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_embedding_model():
    """Download the embedding model used by RAGEngine"""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    logger.info(f"Downloading {model_name}")
    
    try:
        model = SentenceTransformer(model_name)
        logger.info("Model downloaded successfully")
        
        test_embedding = model.encode("test")
        logger.info(f"Model test successful (embedding dimension: {len(test_embedding)})")
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise

if __name__ == "__main__":
    download_embedding_model()
    logger.info("Ready to start the server")
