"""
FAISS index management for efficient similarity search
"""
import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class FAISSIndex:
    """
    Wrapper for FAISS index with persistence and GPU support
    """
    
    def __init__(self,
                 dimension: int,
                 index_type: str = "IndexFlatL2",
                 use_gpu: bool = False,
                 gpu_id: int = 0):
        """
        Initialize FAISS index
        
        Args:
            dimension: Embedding dimension
            index_type: Type of FAISS index
                - "IndexFlatL2": Exact L2 search (best for medical)
                - "IndexFlatIP": Exact inner product search
                - "IndexIVFFlat": Inverted file with exact post-verification
                - "IndexIVFPQ": Inverted file with product quantization
            use_gpu: Whether to use GPU acceleration
            gpu_id: GPU device ID
        """
        self.dimension = dimension
        self.index_type = index_type
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        self.gpu_id = gpu_id
        
        # Create index
        self.index = self._create_index()
        
        # ID mapping (FAISS index -> original ID)
        self.id_to_idx = {}
        self.idx_to_id = {}
        
        logger.info(f"Initialized {index_type} with dimension {dimension}")
        if self.use_gpu:
            logger.info(f"Using GPU {gpu_id}")
    
    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on type"""
        if self.index_type == "IndexFlatL2":
            index = faiss.IndexFlatL2(self.dimension)
            
        elif self.index_type == "IndexFlatIP":
            index = faiss.IndexFlatIP(self.dimension)
            
        elif self.index_type == "IndexIVFFlat":
            # Need to specify nlist (number of clusters)
            nlist = 1000  # Adjust based on dataset size
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            
        elif self.index_type == "IndexIVFPQ":
            # Product quantization for compression
            nlist = 4096
            m = 64  # Number of subquantizers
            nbits = 8  # Bits per subquantizer
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, nbits)
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Move to GPU if requested
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, self.gpu_id, index)
        
        return index
    
    def add(self, embeddings: np.ndarray, ids: Optional[List[str]] = None):
        """
        Add embeddings to index
        
        Args:
            embeddings: Array of shape (N, dimension)
            ids: Optional list of IDs for the embeddings
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        # Ensure float32
        embeddings = embeddings.astype('float32')
        
        # Check dimension
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} != index dimension {self.dimension}")
        
        # Train index if needed (for IVF indexes)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            logger.info("Training index...")
            self.index.train(embeddings)
        
        # Add to index
        start_idx = self.index.ntotal
        self.index.add(embeddings)
        
        # Update ID mapping
        if ids is None:
            ids = [str(start_idx + i) for i in range(len(embeddings))]
        
        for i, uid in enumerate(ids):
            idx = start_idx + i
            self.id_to_idx[uid] = idx
            self.idx_to_id[idx] = uid
        
        logger.info(f"Added {len(embeddings)} embeddings. Total: {self.index.ntotal}")
    
    def search(self, 
               query_embeddings: np.ndarray,
               k: int = 10,
               return_distances: bool = True) -> Tuple[List[List[str]], Optional[np.ndarray]]:
        """
        Search for similar embeddings
        
        Args:
            query_embeddings: Query embedding(s) of shape (N, dimension)
            k: Number of nearest neighbors to return
            return_distances: Whether to return distances
            
        Returns:
            (ids, distances) where:
                - ids: List of lists of neighbor IDs
                - distances: Array of distances (if return_distances=True)
        """
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        # Ensure float32
        query_embeddings = query_embeddings.astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embeddings, k)
        
        # Convert indices to IDs
        result_ids = []
        for idx_list in indices:
            ids = [self.idx_to_id.get(int(idx), None) for idx in idx_list]
            # Filter out None (invalid indices)
            ids = [uid for uid in ids if uid is not None]
            result_ids.append(ids)
        
        if return_distances:
            return result_ids, distances
        else:
            return result_ids, None
    
    def remove(self, ids: List[str]):
        """
        Remove embeddings by ID
        Note: Not all index types support removal
        """
        if not hasattr(self.index, 'remove_ids'):
            logger.warning(f"{self.index_type} does not support removal")
            return
        
        # Convert IDs to indices
        indices = [self.id_to_idx.get(uid) for uid in ids if uid in self.id_to_idx]
        indices = [idx for idx in indices if idx is not None]
        
        if not indices:
            return
        
        # Remove from index
        indices_array = np.array(indices, dtype='int64')
        self.index.remove_ids(indices_array)
        
        # Update mappings
        for uid in ids:
            if uid in self.id_to_idx:
                idx = self.id_to_idx[uid]
                del self.id_to_idx[uid]
                del self.idx_to_id[idx]
        
        logger.info(f"Removed {len(indices)} embeddings")
    
    def save(self, path: str):
        """
        Save index to disk
        
        Args:
            path: Path to save index
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move to CPU if on GPU
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        else:
            cpu_index = self.index
        
        # Save index
        faiss.write_index(cpu_index, str(path))
        
        # Save ID mappings
        mappings = {
            'id_to_idx': self.id_to_idx,
            'idx_to_id': self.idx_to_id
        }
        with open(str(path) + '.mappings', 'wb') as f:
            pickle.dump(mappings, f)
        
        logger.info(f"Saved index to {path}")
    
    def load(self, path: str):
        """
        Load index from disk
        
        Args:
            path: Path to load index from
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")
        
        # Load index
        cpu_index = faiss.read_index(str(path))
        
        # Move to GPU if requested
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, self.gpu_id, cpu_index)
        else:
            self.index = cpu_index
        
        # Load ID mappings
        mappings_path = str(path) + '.mappings'
        if Path(mappings_path).exists():
            with open(mappings_path, 'rb') as f:
                mappings = pickle.load(f)
                self.id_to_idx = mappings['id_to_idx']
                self.idx_to_id = mappings['idx_to_id']
        
        logger.info(f"Loaded index from {path}. Total vectors: {self.index.ntotal}")
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        return {
            'index_type': self.index_type,
            'dimension': self.dimension,
            'total_vectors': self.index.ntotal,
            'use_gpu': self.use_gpu,
            'is_trained': getattr(self.index, 'is_trained', True)
        }


class MultiIndexManager:
    """
    Manage multiple FAISS indexes (e.g., separate indexes for images and text)
    """
    
    def __init__(self):
        """Initialize multi-index manager"""
        self.indexes = {}
    
    def add_index(self, name: str, index: FAISSIndex):
        """Add an index"""
        self.indexes[name] = index
        logger.info(f"Added index: {name}")
    
    def search_all(self, 
                   query_embeddings: Dict[str, np.ndarray],
                   k: int = 10,
                   weights: Optional[Dict[str, float]] = None) -> List[Tuple[str, float]]:
        """
        Search across all indexes and combine results
        
        Args:
            query_embeddings: Dict of {index_name: query_embedding}
            k: Number of results per index
            weights: Weights for each index
            
        Returns:
            Combined and reranked results
        """
        if weights is None:
            weights = {name: 1.0 for name in self.indexes.keys()}
        
        all_results = {}
        
        # Search each index
        for name, index in self.indexes.items():
            if name not in query_embeddings:
                continue
            
            query_emb = query_embeddings[name]
            ids, distances = index.search(query_emb, k=k)
            
            weight = weights.get(name, 1.0)
            
            # Store results with weighted scores
            for uid, dist in zip(ids[0], distances[0]):
                # Convert L2 distance to similarity score
                score = 1.0 / (1.0 + dist)
                
                if uid in all_results:
                    all_results[uid] += weight * score
                else:
                    all_results[uid] = weight * score
        
        # Sort by combined score
        sorted_results = sorted(
            all_results.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results[:k]
    
    def save_all(self, directory: str):
        """Save all indexes"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        for name, index in self.indexes.items():
            index.save(str(directory / f"{name}.index"))
    
    def load_all(self, directory: str):
        """Load all indexes from directory"""
        directory = Path(directory)
        
        for index_file in directory.glob("*.index"):
            name = index_file.stem
            if name in self.indexes:
                self.indexes[name].load(str(index_file))


def test_faiss_index():
    """Test FAISS index"""
    print("=== Testing FAISS Index ===\n")
    
    # Create index
    dimension = 512
    index = FAISSIndex(dimension=dimension, index_type="IndexFlatL2", use_gpu=False)
    
    # Add some vectors
    num_vectors = 1000
    embeddings = np.random.randn(num_vectors, dimension).astype('float32')
    ids = [f"img_{i}" for i in range(num_vectors)]
    
    index.add(embeddings, ids)
    print(f"Added {num_vectors} vectors")
    
    # Search
    query = np.random.randn(1, dimension).astype('float32')
    result_ids, distances = index.search(query, k=10)
    
    print(f"\nTop 10 results:")
    for uid, dist in zip(result_ids[0], distances[0]):
        print(f"  {uid}: {dist:.4f}")
    
    # Save and load
    print("\nTesting save/load...")
    index.save("test_index.faiss")
    
    new_index = FAISSIndex(dimension=dimension, index_type="IndexFlatL2", use_gpu=False)
    new_index.load("test_index.faiss")
    
    print(f"Loaded index with {new_index.index.ntotal} vectors")
    
    # Verify search works after loading
    result_ids2, distances2 = new_index.search(query, k=10)
    assert result_ids == result_ids2, "Search results don't match after loading"
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_faiss_index()
