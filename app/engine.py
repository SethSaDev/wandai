from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .models import RAGResponse
from .utils import (
    generate_cache_key,
    extract_json_from_text,
    create_no_documents_response,
    create_no_results_response,
    filter_relevant_documents,
    build_rag_prompt
)
from cachetools import TTLCache
import logging
import time
import os
import torch

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        self.llm = ChatOllama(
            model="llama3.2:3b",
            temperature=0,
            num_ctx=1024,
            num_predict=256,
            num_gpu=1,
            num_thread=8,
            keep_alive="30m"
        )
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        if device == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vector_store_path = "./vectorstore/faiss_index"
        self.db = None
        
        if os.path.exists(self.vector_store_path):
            try:
                self.db = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                doc_count = self.db.index.ntotal
                logger.info(f"RAGEngine initialized: {doc_count} documents indexed")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")
                self.db = None
        
        self.response_cache = TTLCache(maxsize=100, ttl=1800)
        self.embedding_cache = TTLCache(maxsize=200, ttl=7200)
        
        try:
            self.llm.invoke("test")
        except Exception as e:
            logger.error(f"Model pre-warming failed: {e}")

    def add_document(self, file_path: str):
        try:
            total_start = time.time()
            
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = splitter.split_documents(pages)
            
            embed_start = time.time()
            if self.db is None:
                self.db = FAISS.from_documents(chunks, self.embeddings)
            else:
                self.db.add_documents(chunks)
            embed_time = time.time() - embed_start
            
            os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
            self.db.save_local(self.vector_store_path)
            
            logger.info(f"Indexed {len(chunks)} chunks from {os.path.basename(file_path)}")
        except Exception as e:
            logger.error(f"Failed to index {file_path}: {e}", exc_info=True)
            raise


    async def get_answer(self, question: str):
        if self.db is None:
            return create_no_documents_response()
        
        response_cache_key = generate_cache_key(question)
        if response_cache_key in self.response_cache:
            return self.response_cache[response_cache_key]
        
        try:
            if question in self.embedding_cache:
                query_embedding = self.embedding_cache[question]
            else:
                query_embedding = self.embeddings.embed_query(question)
                self.embedding_cache[question] = query_embedding
            
            docs_with_scores = self.db.similarity_search_with_score_by_vector(
                query_embedding, k=8
            )
            
            if not docs_with_scores:
                logger.warning("No documents found in vector search")
                return create_no_results_response()
            
            relevant_docs = filter_relevant_documents(docs_with_scores)
            docs = [doc for doc, _ in relevant_docs]
            context = "\n\n".join([d.page_content for d in docs])
            
            final_prompt = build_rag_prompt(context, question)
            
            llm_start = time.time()
            response = await self.llm.ainvoke(final_prompt)
            llm_time = time.time() - llm_start
            
            content = extract_json_from_text(response.content)
            
            if llm_time > 15:
                logger.warning(f"Slow LLM inference: {llm_time:.1f}s")
            
            self.response_cache[response_cache_key] = content
            return content
            
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            raise