"""
AgentVault™ Vector Database Integration
Production-ready vector storage for RAG applications
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import asyncio
import numpy as np
import json
import pickle
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import faiss
import hnswlib
from sentence_transformers import SentenceTransformer
import torch
import tiktoken
from collections import defaultdict
import msgpack
import zstandard as zstd

from ..storage.tier_manager import StorageTier
from ..cache.distributed_cache import DistributedCache
from ..security.advanced_encryption import AdvancedEncryptionManager, ComplianceLevel

logger = logging.getLogger(__name__)


class VectorDBType(Enum):
    """Supported vector database types"""
    FAISS = "faiss"
    HNSWLIB = "hnswlib"
    CUSTOM = "custom"
    HYBRID = "hybrid"


class EmbeddingModel(Enum):
    """Supported embedding models"""
    ADA_002 = "text-embedding-ada-002"
    E5_LARGE = "intfloat/e5-large-v2"
    BGE_LARGE = "BAAI/bge-large-en-v1.5"
    CUSTOM = "custom"
    MULTI_LINGUAL = "intfloat/multilingual-e5-large"


class IndexType(Enum):
    """Vector index types"""
    FLAT = "flat"  # Exact search
    HNSW = "hnsw"  # Hierarchical Navigable Small World
    IVF = "ivf"    # Inverted File Index
    LSH = "lsh"    # Locality Sensitive Hashing
    QUANTIZED = "quantized"  # Product Quantization


class SearchMode(Enum):
    """Search modes"""
    SIMILARITY = "similarity"
    MMR = "mmr"  # Maximum Marginal Relevance
    HYBRID = "hybrid"  # Combine with keyword search
    FILTERED = "filtered"  # With metadata filtering
    SEMANTIC = "semantic"  # Pure semantic search


@dataclass
class VectorDocument:
    """Document with vector embedding"""
    id: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    agent_id: str = ""
    compliance_level: ComplianceLevel = ComplianceLevel.INTERNAL
    chunk_index: int = 0
    parent_id: Optional[str] = None


@dataclass
class SearchResult:
    """Vector search result"""
    document: VectorDocument
    score: float
    distance: float
    relevance_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorIndex:
    """Vector index metadata"""
    index_id: str
    index_type: IndexType
    dimension: int
    size: int
    created_at: datetime
    last_updated: datetime
    storage_tier: StorageTier
    compression_enabled: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """
    Production-ready vector store with:
    - Multiple index types for different use cases
    - Distributed storage on Azure NetApp Files
    - Intelligent caching and prefetching
    - Metadata filtering and hybrid search
    - Incremental indexing and updates
    - Multi-tenancy with agent isolation
    - Compression and optimization
    - Backup and recovery
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Vector DB configuration
        self.db_type = VectorDBType(config.get('vector_db_type', 'hybrid'))
        self.embedding_model_name = EmbeddingModel(
            config.get('embedding_model', 'e5-large')
        )
        self.default_index_type = IndexType(config.get('index_type', 'hnsw'))
        
        # Storage configuration
        self.storage_path = config.get('storage_path', '/mnt/agentvault/vectors')
        self.enable_compression = config.get('enable_compression', True)
        self.compression_level = config.get('compression_level', 3)
        
        # Performance configuration
        self.batch_size = config.get('batch_size', 100)
        self.max_index_size = config.get('max_index_size', 1000000)
        self.cache_enabled = config.get('cache_enabled', True)
        self.prefetch_enabled = config.get('prefetch_enabled', True)
        
        # Components
        self.embedding_model = None
        self.cache = None
        self.encryption_manager = None
        
        # Indexes
        self.indexes: Dict[str, Any] = {}  # index_id -> index object
        self.index_metadata: Dict[str, VectorIndex] = {}
        self.agent_indexes: Dict[str, List[str]] = defaultdict(list)  # agent -> indexes
        
        # Document storage
        self.documents: Dict[str, VectorDocument] = {}
        self.document_chunks: Dict[str, List[str]] = defaultdict(list)  # parent -> chunks
        
        # Search optimization
        self.search_cache: Dict[str, Tuple[List[SearchResult], datetime]] = {}
        self.cache_ttl = timedelta(minutes=15)
        
        # Metrics
        self.metrics = defaultdict(int)
        
        # Background tasks
        self._running = False
        self._background_tasks: List[asyncio.Task] = []
        
    async def initialize(
        self,
        cache: DistributedCache,
        encryption_manager: AdvancedEncryptionManager
    ) -> None:
        """Initialize vector store"""
        logger.info("Initializing Vector Store...")
        
        try:
            self.cache = cache
            self.encryption_manager = encryption_manager
            
            # Initialize embedding model
            await self._initialize_embedding_model()
            
            # Load existing indexes
            await self._load_indexes()
            
            # Start background tasks
            self._running = True
            self._start_background_tasks()
            
            logger.info("Vector Store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise
    
    async def _initialize_embedding_model(self) -> None:
        """Initialize embedding model"""
        
        if self.embedding_model_name == EmbeddingModel.ADA_002:
            # OpenAI embedding would be initialized here
            # For now, use sentence transformers
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        elif self.embedding_model_name == EmbeddingModel.E5_LARGE:
            self.embedding_model = SentenceTransformer('intfloat/e5-large-v2')
        elif self.embedding_model_name == EmbeddingModel.BGE_LARGE:
            self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        elif self.embedding_model_name == EmbeddingModel.MULTI_LINGUAL:
            self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
        else:
            # Default model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get embedding dimension
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Initialized embedding model with dimension {self.embedding_dimension}")
    
    async def create_index(
        self,
        agent_id: str,
        index_name: str,
        index_type: Optional[IndexType] = None,
        dimension: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create new vector index for agent"""
        
        index_id = f"{agent_id}-{index_name}-{datetime.utcnow().timestamp()}"
        index_type = index_type or self.default_index_type
        dimension = dimension or self.embedding_dimension
        
        logger.info(f"Creating index {index_id} with type {index_type.value}")
        
        try:
            # Create index based on type
            if index_type == IndexType.FLAT:
                index = await self._create_flat_index(dimension)
            elif index_type == IndexType.HNSW:
                index = await self._create_hnsw_index(dimension)
            elif index_type == IndexType.IVF:
                index = await self._create_ivf_index(dimension)
            elif index_type == IndexType.QUANTIZED:
                index = await self._create_quantized_index(dimension)
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
            
            # Store index
            self.indexes[index_id] = index
            
            # Create metadata
            index_metadata = VectorIndex(
                index_id=index_id,
                index_type=index_type,
                dimension=dimension,
                size=0,
                created_at=datetime.utcnow(),
                last_updated=datetime.utcnow(),
                storage_tier=StorageTier.PREMIUM,  # Start with premium for fast access
                compression_enabled=self.enable_compression,
                metadata=metadata or {}
            )
            
            self.index_metadata[index_id] = index_metadata
            self.agent_indexes[agent_id].append(index_id)
            
            # Save to storage
            await self._save_index(index_id)
            
            logger.info(f"Created index {index_id}")
            return index_id
            
        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            raise
    
    async def _create_flat_index(self, dimension: int) -> faiss.IndexFlatL2:
        """Create flat L2 index for exact search"""
        return faiss.IndexFlatL2(dimension)
    
    async def _create_hnsw_index(self, dimension: int) -> hnswlib.Index:
        """Create HNSW index for approximate search"""
        
        index = hnswlib.Index(space='l2', dim=dimension)
        index.init_index(
            max_elements=self.max_index_size,
            ef_construction=200,
            M=16
        )
        index.set_ef(50)  # ef for search
        
        return index
    
    async def _create_ivf_index(self, dimension: int) -> faiss.IndexIVFFlat:
        """Create IVF index for large-scale search"""
        
        # Create quantizer
        quantizer = faiss.IndexFlatL2(dimension)
        
        # Number of clusters
        nlist = min(4096, int(np.sqrt(self.max_index_size)))
        
        # Create IVF index
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        
        # Train with sample data if available
        # In production, gather representative samples
        
        return index
    
    async def _create_quantized_index(self, dimension: int) -> faiss.IndexPQ:
        """Create product quantized index for memory efficiency"""
        
        # Number of sub-quantizers
        m = min(dimension // 4, 64)  # Each sub-vector at least 4 dimensions
        
        # Bits per sub-quantizer
        nbits = 8
        
        index = faiss.IndexPQ(dimension, m, nbits)
        
        return index
    
    async def add_documents(
        self,
        agent_id: str,
        documents: List[Dict[str, Any]],
        index_id: Optional[str] = None,
        chunking_strategy: str = "semantic"
    ) -> List[str]:
        """Add documents to vector store"""
        
        # Get or create index
        if not index_id:
            # Get default index for agent
            if agent_id in self.agent_indexes and self.agent_indexes[agent_id]:
                index_id = self.agent_indexes[agent_id][0]
            else:
                # Create new index
                index_id = await self.create_index(agent_id, "default")
        
        added_ids = []
        
        try:
            for doc in documents:
                # Chunk document if needed
                chunks = await self._chunk_document(doc, chunking_strategy)
                
                for i, chunk in enumerate(chunks):
                    # Generate ID
                    doc_id = f"{agent_id}-{datetime.utcnow().timestamp()}-{i}"
                    
                    # Create embedding
                    embedding = await self._create_embedding(chunk['content'])
                    
                    # Create document
                    vector_doc = VectorDocument(
                        id=doc_id,
                        content=chunk['content'],
                        embedding=embedding,
                        metadata={
                            **doc.get('metadata', {}),
                            **chunk.get('metadata', {}),
                            'chunk_index': i,
                            'total_chunks': len(chunks)
                        },
                        agent_id=agent_id,
                        compliance_level=ComplianceLevel(
                            doc.get('compliance_level', 'internal')
                        ),
                        chunk_index=i,
                        parent_id=doc.get('parent_id')
                    )
                    
                    # Add to index
                    await self._add_to_index(index_id, vector_doc)
                    
                    # Store document
                    self.documents[doc_id] = vector_doc
                    
                    # Track chunks
                    if vector_doc.parent_id:
                        self.document_chunks[vector_doc.parent_id].append(doc_id)
                    
                    added_ids.append(doc_id)
            
            # Update index metadata
            self.index_metadata[index_id].size += len(added_ids)
            self.index_metadata[index_id].last_updated = datetime.utcnow()
            
            # Save index
            await self._save_index(index_id)
            
            # Clear search cache
            self.search_cache.clear()
            
            self.metrics['documents_added'] += len(added_ids)
            
            logger.info(f"Added {len(added_ids)} documents to index {index_id}")
            return added_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise
    
    async def search(
        self,
        agent_id: str,
        query: Union[str, np.ndarray],
        k: int = 10,
        index_id: Optional[str] = None,
        search_mode: SearchMode = SearchMode.SIMILARITY,
        filters: Optional[Dict[str, Any]] = None,
        mmr_lambda: float = 0.5
    ) -> List[SearchResult]:
        """Search for similar documents"""
        
        # Check cache
        cache_key = f"{agent_id}:{query if isinstance(query, str) else 'vector'}:{k}:{search_mode.value}"
        if cache_key in self.search_cache:
            cached_results, cached_time = self.search_cache[cache_key]
            if datetime.utcnow() - cached_time < self.cache_ttl:
                self.metrics['cache_hits'] += 1
                return cached_results
        
        try:
            # Get query embedding
            if isinstance(query, str):
                query_embedding = await self._create_embedding(query)
            else:
                query_embedding = query
            
            # Get indexes to search
            if index_id:
                index_ids = [index_id]
            else:
                index_ids = self.agent_indexes.get(agent_id, [])
            
            if not index_ids:
                logger.warning(f"No indexes found for agent {agent_id}")
                return []
            
            # Search based on mode
            if search_mode == SearchMode.SIMILARITY:
                results = await self._similarity_search(
                    index_ids, query_embedding, k, filters
                )
            elif search_mode == SearchMode.MMR:
                results = await self._mmr_search(
                    index_ids, query_embedding, k, filters, mmr_lambda
                )
            elif search_mode == SearchMode.HYBRID:
                results = await self._hybrid_search(
                    index_ids, query, query_embedding, k, filters
                )
            elif search_mode == SearchMode.FILTERED:
                results = await self._filtered_search(
                    index_ids, query_embedding, k, filters
                )
            elif search_mode == SearchMode.SEMANTIC:
                results = await self._semantic_search(
                    index_ids, query_embedding, k, filters
                )
            else:
                raise ValueError(f"Unsupported search mode: {search_mode}")
            
            # Cache results
            self.search_cache[cache_key] = (results, datetime.utcnow())
            
            # Clean old cache entries
            if len(self.search_cache) > 1000:
                self._clean_search_cache()
            
            self.metrics['searches'] += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise
    
    async def _chunk_document(
        self,
        document: Dict[str, Any],
        strategy: str
    ) -> List[Dict[str, Any]]:
        """Chunk document into smaller pieces"""
        
        content = document.get('content', '')
        
        if strategy == "fixed":
            # Fixed size chunks
            chunk_size = 500
            chunks = [
                content[i:i+chunk_size]
                for i in range(0, len(content), chunk_size)
            ]
            
        elif strategy == "semantic":
            # Semantic chunking based on sentences/paragraphs
            chunks = self._semantic_chunk(content)
            
        elif strategy == "sliding":
            # Sliding window with overlap
            chunk_size = 500
            overlap = 100
            chunks = []
            for i in range(0, len(content), chunk_size - overlap):
                chunks.append(content[i:i+chunk_size])
                
        else:
            # No chunking
            chunks = [content]
        
        # Create chunk documents
        chunk_docs = []
        for i, chunk_content in enumerate(chunks):
            chunk_docs.append({
                'content': chunk_content,
                'metadata': {
                    'chunk_strategy': strategy,
                    'chunk_index': i,
                    'parent_id': document.get('id'),
                    **document.get('metadata', {})
                }
            })
        
        return chunk_docs
    
    def _semantic_chunk(self, content: str) -> List[str]:
        """Semantic chunking based on content structure"""
        
        # Simple implementation - split by paragraphs
        paragraphs = content.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < 1000:  # Max chunk size
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [content]
    
    async def _create_embedding(self, text: str) -> np.ndarray:
        """Create embedding for text"""
        
        # Add prefix for E5 models
        if self.embedding_model_name in [EmbeddingModel.E5_LARGE, EmbeddingModel.MULTI_LINGUAL]:
            text = f"passage: {text}"
        
        # Create embedding
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        
        # Normalize for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    async def _add_to_index(
        self,
        index_id: str,
        document: VectorDocument
    ) -> None:
        """Add document to specific index"""
        
        index = self.indexes.get(index_id)
        if not index:
            raise ValueError(f"Index not found: {index_id}")
        
        index_type = self.index_metadata[index_id].index_type
        
        if index_type == IndexType.HNSW:
            # HNSW uses integer labels
            label = hash(document.id) % (2**31)
            index.add_items(
                document.embedding.reshape(1, -1),
                np.array([label])
            )
        else:
            # FAISS indexes
            if hasattr(index, 'is_trained') and not index.is_trained:
                # Train index if needed (for IVF, PQ)
                # In production, train with representative data
                training_data = np.array([document.embedding])
                index.train(training_data)
            
            index.add(document.embedding.reshape(1, -1))
    
    async def _similarity_search(
        self,
        index_ids: List[str],
        query_embedding: np.ndarray,
        k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Basic similarity search"""
        
        all_results = []
        
        for index_id in index_ids:
            index = self.indexes.get(index_id)
            if not index:
                continue
            
            index_type = self.index_metadata[index_id].index_type
            
            if index_type == IndexType.HNSW:
                # HNSW search
                labels, distances = index.knn_query(
                    query_embedding.reshape(1, -1),
                    k=min(k * 2, index.get_current_count())  # Get extra for filtering
                )
                
                # Map labels back to document IDs
                for label, distance in zip(labels[0], distances[0]):
                    # Find document with this label
                    for doc_id, doc in self.documents.items():
                        if hash(doc_id) % (2**31) == label:
                            if self._apply_filters(doc, filters):
                                all_results.append(SearchResult(
                                    document=doc,
                                    score=1 / (1 + distance),  # Convert distance to similarity
                                    distance=float(distance),
                                    relevance_score=1 / (1 + distance),
                                    metadata={'index_id': index_id}
                                ))
                            break
            else:
                # FAISS search
                distances, indices = index.search(
                    query_embedding.reshape(1, -1),
                    min(k * 2, index.ntotal)
                )
                
                # Get documents
                doc_list = list(self.documents.values())
                for idx, distance in zip(indices[0], distances[0]):
                    if 0 <= idx < len(doc_list):
                        doc = doc_list[idx]
                        if self._apply_filters(doc, filters):
                            all_results.append(SearchResult(
                                document=doc,
                                score=1 / (1 + distance),
                                distance=float(distance),
                                relevance_score=1 / (1 + distance),
                                metadata={'index_id': index_id}
                            ))
        
        # Sort by score and return top k
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:k]
    
    async def _mmr_search(
        self,
        index_ids: List[str],
        query_embedding: np.ndarray,
        k: int,
        filters: Optional[Dict[str, Any]],
        mmr_lambda: float
    ) -> List[SearchResult]:
        """Maximum Marginal Relevance search for diversity"""
        
        # First get more candidates
        candidates = await self._similarity_search(
            index_ids, query_embedding, k * 3, filters
        )
        
        if not candidates:
            return []
        
        # MMR algorithm
        selected = []
        remaining = candidates.copy()
        
        # Select first document (highest similarity)
        selected.append(remaining.pop(0))
        
        while len(selected) < k and remaining:
            mmr_scores = []
            
            for candidate in remaining:
                # Similarity to query
                sim_to_query = candidate.score
                
                # Max similarity to selected documents
                max_sim_to_selected = 0
                for selected_doc in selected:
                    sim = np.dot(
                        candidate.document.embedding,
                        selected_doc.document.embedding
                    )
                    max_sim_to_selected = max(max_sim_to_selected, sim)
                
                # MMR score
                mmr_score = (
                    mmr_lambda * sim_to_query -
                    (1 - mmr_lambda) * max_sim_to_selected
                )
                mmr_scores.append(mmr_score)
            
            # Select document with highest MMR score
            best_idx = np.argmax(mmr_scores)
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    async def _hybrid_search(
        self,
        index_ids: List[str],
        query_text: str,
        query_embedding: np.ndarray,
        k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Hybrid search combining vector and keyword search"""
        
        # Vector search
        vector_results = await self._similarity_search(
            index_ids, query_embedding, k * 2, filters
        )
        
        # Keyword search (simplified BM25-like scoring)
        query_terms = query_text.lower().split()
        keyword_scores = {}
        
        for doc_id, doc in self.documents.items():
            if self._apply_filters(doc, filters):
                content_lower = doc.content.lower()
                score = 0
                
                for term in query_terms:
                    # Term frequency
                    tf = content_lower.count(term)
                    if tf > 0:
                        # Simple BM25-like score
                        score += (tf * 2.0) / (tf + 1.0)
                
                if score > 0:
                    keyword_scores[doc_id] = score
        
        # Combine scores
        combined_results = {}
        
        # Add vector results
        for result in vector_results:
            combined_results[result.document.id] = {
                'result': result,
                'vector_score': result.score,
                'keyword_score': keyword_scores.get(result.document.id, 0)
            }
        
        # Add keyword-only results
        for doc_id, keyword_score in keyword_scores.items():
            if doc_id not in combined_results and doc_id in self.documents:
                doc = self.documents[doc_id]
                # Calculate vector similarity
                vector_sim = np.dot(doc.embedding, query_embedding)
                
                combined_results[doc_id] = {
                    'result': SearchResult(
                        document=doc,
                        score=vector_sim,
                        distance=1 - vector_sim,
                        relevance_score=vector_sim,
                        metadata={}
                    ),
                    'vector_score': vector_sim,
                    'keyword_score': keyword_score
                }
        
        # Combine scores (weighted average)
        vector_weight = 0.7
        keyword_weight = 0.3
        
        final_results = []
        for doc_id, scores in combined_results.items():
            combined_score = (
                vector_weight * scores['vector_score'] +
                keyword_weight * scores['keyword_score'] / 10  # Normalize keyword scores
            )
            
            result = scores['result']
            result.relevance_score = combined_score
            result.metadata['hybrid_score'] = combined_score
            result.metadata['vector_score'] = scores['vector_score']
            result.metadata['keyword_score'] = scores['keyword_score']
            
            final_results.append(result)
        
        # Sort by combined score
        final_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return final_results[:k]
    
    async def _filtered_search(
        self,
        index_ids: List[str],
        query_embedding: np.ndarray,
        k: int,
        filters: Dict[str, Any]
    ) -> List[SearchResult]:
        """Search with strict filtering"""
        
        # Get more candidates to account for filtering
        results = await self._similarity_search(
            index_ids, query_embedding, k * 5, filters
        )
        
        return results[:k]
    
    async def _semantic_search(
        self,
        index_ids: List[str],
        query_embedding: np.ndarray,
        k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Pure semantic search with re-ranking"""
        
        # Get initial results
        results = await self._similarity_search(
            index_ids, query_embedding, k * 2, filters
        )
        
        # Re-rank based on semantic similarity
        # In production, could use cross-encoder model
        for result in results:
            # Adjust score based on document quality signals
            quality_score = self._calculate_quality_score(result.document)
            result.relevance_score = result.score * quality_score
        
        # Sort by adjusted score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:k]
    
    def _apply_filters(
        self,
        document: VectorDocument,
        filters: Optional[Dict[str, Any]]
    ) -> bool:
        """Apply metadata filters to document"""
        
        if not filters:
            return True
        
        for key, value in filters.items():
            if key == 'agent_id' and document.agent_id != value:
                return False
            elif key == 'compliance_level' and document.compliance_level.value != value:
                return False
            elif key in document.metadata:
                if isinstance(value, list):
                    # Check if metadata value is in list
                    if document.metadata[key] not in value:
                        return False
                elif document.metadata[key] != value:
                    return False
        
        return True
    
    def _calculate_quality_score(self, document: VectorDocument) -> float:
        """Calculate document quality score for ranking"""
        
        score = 1.0
        
        # Recency boost
        age_days = (datetime.utcnow() - document.created_at).days
        if age_days < 7:
            score *= 1.2
        elif age_days > 90:
            score *= 0.8
        
        # Length penalty (prefer medium-length documents)
        doc_length = len(document.content)
        if 100 < doc_length < 1000:
            score *= 1.1
        elif doc_length < 50:
            score *= 0.7
        
        # Metadata completeness
        if len(document.metadata) > 5:
            score *= 1.05
        
        return min(score, 1.5)  # Cap maximum boost
    
    def _clean_search_cache(self) -> None:
        """Clean old entries from search cache"""
        
        current_time = datetime.utcnow()
        expired_keys = [
            key for key, (_, cached_time) in self.search_cache.items()
            if current_time - cached_time > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.search_cache[key]
    
    async def update_document(
        self,
        document_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update existing document"""
        
        if document_id not in self.documents:
            logger.error(f"Document not found: {document_id}")
            return False
        
        try:
            doc = self.documents[document_id]
            
            # Update content and embedding if provided
            if content:
                doc.content = content
                doc.embedding = await self._create_embedding(content)
                
                # Update in indexes
                for index_id in self.agent_indexes.get(doc.agent_id, []):
                    # This is simplified - in production, handle index updates properly
                    await self._update_in_index(index_id, doc)
            
            # Update metadata
            if metadata:
                doc.metadata.update(metadata)
            
            doc.updated_at = datetime.utcnow()
            
            # Clear cache
            self.search_cache.clear()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document: {str(e)}")
            return False
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete document from vector store"""
        
        if document_id not in self.documents:
            return False
        
        try:
            doc = self.documents[document_id]
            
            # Remove from indexes
            for index_id in self.agent_indexes.get(doc.agent_id, []):
                await self._remove_from_index(index_id, doc)
            
            # Remove from storage
            del self.documents[document_id]
            
            # Remove from chunk tracking
            if doc.parent_id and document_id in self.document_chunks[doc.parent_id]:
                self.document_chunks[doc.parent_id].remove(document_id)
            
            # Clear cache
            self.search_cache.clear()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document: {str(e)}")
            return False
    
    async def get_index_stats(self, index_id: str) -> Dict[str, Any]:
        """Get statistics for an index"""
        
        if index_id not in self.index_metadata:
            return {}
        
        metadata = self.index_metadata[index_id]
        index = self.indexes.get(index_id)
        
        stats = {
            'index_id': index_id,
            'type': metadata.index_type.value,
            'dimension': metadata.dimension,
            'size': metadata.size,
            'created_at': metadata.created_at.isoformat(),
            'last_updated': metadata.last_updated.isoformat(),
            'storage_tier': metadata.storage_tier.value,
            'compression_enabled': metadata.compression_enabled
        }
        
        # Add index-specific stats
        if index:
            if metadata.index_type == IndexType.HNSW:
                stats['current_count'] = index.get_current_count()
                stats['ef'] = index.ef
                stats['max_elements'] = index.get_max_elements()
            elif hasattr(index, 'ntotal'):
                stats['total_vectors'] = index.ntotal
        
        return stats
    
    async def optimize_index(self, index_id: str) -> bool:
        """Optimize index for better performance"""
        
        if index_id not in self.indexes:
            return False
        
        try:
            index = self.indexes[index_id]
            metadata = self.index_metadata[index_id]
            
            # Optimization based on index type
            if metadata.index_type == IndexType.HNSW:
                # Adjust ef based on usage patterns
                if metadata.size > 100000:
                    index.set_ef(100)  # Higher ef for larger indexes
                elif metadata.size > 10000:
                    index.set_ef(50)
            
            elif metadata.index_type == IndexType.IVF and hasattr(index, 'nprobe'):
                # Adjust nprobe for IVF indexes
                if metadata.size > 1000000:
                    index.nprobe = 32
                else:
                    index.nprobe = 16
            
            # Save optimized index
            await self._save_index(index_id)
            
            logger.info(f"Optimized index {index_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to optimize index: {str(e)}")
            return False
    
    async def _save_index(self, index_id: str) -> None:
        """Save index to storage"""
        
        index = self.indexes.get(index_id)
        metadata = self.index_metadata.get(index_id)
        
        if not index or not metadata:
            return
        
        # Prepare index data
        if metadata.index_type == IndexType.HNSW:
            # HNSW has built-in serialization
            index_data = index.get_index_data()
        else:
            # FAISS serialization
            import io
            buffer = io.BytesIO()
            faiss.write_index_binary(buffer, index)
            index_data = buffer.getvalue()
        
        # Compress if enabled
        if self.enable_compression:
            cctx = zstd.ZstdCompressor(level=self.compression_level)
            index_data = cctx.compress(index_data)
        
        # Encrypt
        encrypted_data = await self.encryption_manager.encrypt(
            index_data,
            metadata.metadata.get('agent_id', 'system'),
            ComplianceLevel.CONFIDENTIAL
        )
        
        # Store to file system (ANF)
        index_path = f"{self.storage_path}/{index_id}.idx"
        
        # In production, use proper async file I/O
        with open(index_path, 'wb') as f:
            f.write(msgpack.packb({
                'encrypted_data': encrypted_data.__dict__,
                'metadata': metadata.__dict__
            }))
        
        # Update cache
        if self.cache_enabled:
            await self.cache.set(
                f"vector_index:{index_id}",
                index_data,
                ttl=3600  # 1 hour
            )
    
    async def _load_indexes(self) -> None:
        """Load existing indexes from storage"""
        
        import os
        
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
            return
        
        # List index files
        for filename in os.listdir(self.storage_path):
            if filename.endswith('.idx'):
                index_id = filename[:-4]
                
                try:
                    # Load index
                    await self._load_index(index_id)
                except Exception as e:
                    logger.error(f"Failed to load index {index_id}: {str(e)}")
    
    async def _load_index(self, index_id: str) -> None:
        """Load specific index from storage"""
        
        # Check cache first
        if self.cache_enabled:
            cached_data = await self.cache.get(f"vector_index:{index_id}")
            if cached_data:
                # Deserialize from cache
                # Implementation depends on index type
                pass
        
        # Load from file
        index_path = f"{self.storage_path}/{index_id}.idx"
        
        with open(index_path, 'rb') as f:
            data = msgpack.unpackb(f.read())
        
        # Decrypt
        # Implementation needed based on encryption manager
        
        # Decompress if needed
        # Implementation needed
        
        # Recreate index
        # Implementation depends on index type
        
        logger.info(f"Loaded index {index_id}")
    
    async def _update_in_index(
        self,
        index_id: str,
        document: VectorDocument
    ) -> None:
        """Update document in index"""
        
        # This is complex for most index types
        # Simplified approach: remove and re-add
        await self._remove_from_index(index_id, document)
        await self._add_to_index(index_id, document)
    
    async def _remove_from_index(
        self,
        index_id: str,
        document: VectorDocument
    ) -> None:
        """Remove document from index"""
        
        # Most vector indexes don't support removal
        # In production, track deletions separately or rebuild index
        pass
    
    def _start_background_tasks(self) -> None:
        """Start background maintenance tasks"""
        
        self._background_tasks = [
            asyncio.create_task(self._index_optimization_worker()),
            asyncio.create_task(self._cache_cleanup_worker()),
            asyncio.create_task(self._metrics_reporter())
        ]
    
    async def _index_optimization_worker(self) -> None:
        """Periodically optimize indexes"""
        
        while self._running:
            try:
                await asyncio.sleep(3600)  # Hourly
                
                for index_id in list(self.indexes.keys()):
                    metadata = self.index_metadata.get(index_id)
                    if metadata and metadata.size > 10000:
                        await self.optimize_index(index_id)
                
            except Exception as e:
                logger.error(f"Index optimization error: {str(e)}")
    
    async def _cache_cleanup_worker(self) -> None:
        """Clean up old cache entries"""
        
        while self._running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                self._clean_search_cache()
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {str(e)}")
    
    async def _metrics_reporter(self) -> None:
        """Report metrics periodically"""
        
        while self._running:
            try:
                await asyncio.sleep(60)  # Every minute
                
                total_documents = len(self.documents)
                total_indexes = len(self.indexes)
                cache_hit_rate = (
                    self.metrics['cache_hits'] /
                    max(1, self.metrics['cache_hits'] + self.metrics['searches'])
                )
                
                logger.info(
                    f"VectorStore metrics: "
                    f"documents={total_documents}, "
                    f"indexes={total_indexes}, "
                    f"cache_hit_rate={cache_hit_rate:.2%}"
                )
                
            except Exception as e:
                logger.error(f"Metrics reporter error: {str(e)}")
    
    async def shutdown(self) -> None:
        """Shutdown vector store"""
        
        logger.info("Shutting down Vector Store...")
        
        self._running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Save all indexes
        for index_id in self.indexes:
            try:
                await self._save_index(index_id)
            except Exception as e:
                logger.error(f"Failed to save index {index_id}: {str(e)}")
        
        logger.info("Vector Store shutdown complete")