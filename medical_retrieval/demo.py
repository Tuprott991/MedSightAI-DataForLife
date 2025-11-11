"""
Demo script for medical multimodal retrieval system
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
from PIL import Image
import logging
from typing import List, Dict
import time

from data.dataset_loader import IndianaDatasetLoader
from data.preprocessor import TextPreprocessor
from models.encoder import MedCLIPEncoder
from models.reranker import MedicalReranker
from indexing.faiss_index import FAISSIndex
from indexing.database import MetadataDatabase
from config import (
    MODEL_CONFIG, INDEX_CONFIG, RETRIEVAL_CONFIG,
    RERANKING_WEIGHTS, INDEX_DIR
)
from model_selector import get_active_model_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MedicalRetrievalSystem:
    """Complete medical image retrieval system"""
    
    def __init__(self,
                 index_dir: str,
                 use_gpu: bool = False):
        """
        Initialize retrieval system
        
        Args:
            index_dir: Directory containing indexes and database
            use_gpu: Whether to use GPU
        """
        self.index_dir = Path(index_dir)
        self.use_gpu = use_gpu
        
        logger.info("Initializing Medical Retrieval System...")
        
        # Load encoder
        logger.info("Loading encoder...")
        model_config = get_active_model_config()
        
        logger.info(f"Using model: {model_config['active_model']}")
        
        self.encoder = MedCLIPEncoder(
            model_name=model_config['model_name'],
            model_type=model_config.get('model_type', 'clip'),
            device="cuda" if use_gpu else "cpu"
        )
        
        # Load indexes
        logger.info("Loading FAISS indexes...")
        embedding_dim = self.encoder.get_embedding_dim()
        
        self.image_index = FAISSIndex(
            dimension=embedding_dim,
            index_type=INDEX_CONFIG['type'],
            use_gpu=use_gpu,
            gpu_id=INDEX_CONFIG['gpu_id']
        )
        self.image_index.load(str(self.index_dir / "image_index.faiss"))
        
        self.text_index = FAISSIndex(
            dimension=embedding_dim,
            index_type=INDEX_CONFIG['type'],
            use_gpu=use_gpu,
            gpu_id=INDEX_CONFIG['gpu_id']
        )
        self.text_index.load(str(self.index_dir / "text_index.faiss"))
        
        # Load database
        logger.info("Loading database...")
        self.database = MetadataDatabase(str(self.index_dir / "metadata.db"))
        
        # Initialize reranker
        self.reranker = MedicalReranker(weights=RERANKING_WEIGHTS)
        
        # Text preprocessor
        self.text_preprocessor = TextPreprocessor()
        
        logger.info("✓ System initialized successfully!")
        logger.info(f"  - Image index: {self.image_index.index.ntotal} vectors")
        logger.info(f"  - Text index: {self.text_index.index.ntotal} vectors")
        logger.info(f"  - Database: {self.database.count()} records")
    
    def search_by_text(self,
                      query_text: str,
                      top_k: int = 10,
                      enable_reranking: bool = True) -> List[Dict]:
        """
        Search by text query
        
        Args:
            query_text: Text query
            top_k: Number of results
            enable_reranking: Whether to enable reranking
            
        Returns:
            List of search results
        """
        start_time = time.time()
        
        # Encode query
        query_embedding = self.encoder.encode_text(query_text, normalize=True)[0]
        
        # Search
        top_k_retrieval = RETRIEVAL_CONFIG['top_k_initial'] if enable_reranking else top_k
        result_ids, distances = self.text_index.search(
            query_embedding.reshape(1, -1),
            k=top_k_retrieval
        )
        
        # Get candidates
        candidates = self.database.get_many(result_ids[0])
        
        # Reranking
        if enable_reranking and len(candidates) > 0:
            candidate_embeddings = []
            for candidate in candidates:
                combined_text = self.text_preprocessor.combine_clinical_text(
                    findings=candidate['findings'],
                    impression=candidate['impression']
                )
                txt_emb = self.encoder.encode_text(combined_text, normalize=True)[0]
                candidate_embeddings.append({
                    'findings': txt_emb,
                    'impression': txt_emb,
                    'image': None
                })
            
            query_emb_dict = {
                'findings': query_embedding,
                'impression': query_embedding,
                'image': None
            }
            
            ranked = self.reranker.rerank({}, candidates, query_emb_dict, candidate_embeddings)
            results = [c for c, _ in ranked[:top_k]]
            scores = [s for _, s in ranked[:top_k]]
        else:
            results = candidates[:top_k]
            scores = [1.0 / (1.0 + d) for d in distances[0][:top_k]]
        
        # Add scores to results
        for result, score in zip(results, scores):
            result['score'] = score
        
        elapsed = time.time() - start_time
        logger.info(f"Text search completed in {elapsed*1000:.2f}ms")
        
        return results
    
    def search_by_image(self,
                       query_image: Image.Image,
                       top_k: int = 10) -> List[Dict]:
        """
        Search by image query
        
        Args:
            query_image: PIL Image
            top_k: Number of results
            
        Returns:
            List of search results
        """
        start_time = time.time()
        
        # Encode query
        query_embedding = self.encoder.encode_image(query_image, normalize=True)[0]
        
        # Search
        result_ids, distances = self.image_index.search(
            query_embedding.reshape(1, -1),
            k=top_k
        )
        
        # Get candidates
        results = self.database.get_many(result_ids[0])
        
        # Add scores
        scores = [1.0 / (1.0 + d) for d in distances[0][:top_k]]
        for result, score in zip(results, scores):
            result['score'] = score
        
        elapsed = time.time() - start_time
        logger.info(f"Image search completed in {elapsed*1000:.2f}ms")
        
        return results
    
    def search_multimodal(self,
                         query_text: str,
                         query_image: Image.Image,
                         top_k: int = 10,
                         text_weight: float = 0.5,
                         image_weight: float = 0.5) -> List[Dict]:
        """
        Search by both text and image
        
        Args:
            query_text: Text query
            query_image: PIL Image
            top_k: Number of results
            text_weight: Weight for text similarity
            image_weight: Weight for image similarity
            
        Returns:
            List of search results
        """
        start_time = time.time()
        
        # Encode queries
        text_embedding = self.encoder.encode_text(query_text, normalize=True)[0]
        image_embedding = self.encoder.encode_image(query_image, normalize=True)[0]
        
        # Search both indexes
        img_ids, img_dists = self.image_index.search(image_embedding.reshape(1, -1), k=top_k*2)
        txt_ids, txt_dists = self.text_index.search(text_embedding.reshape(1, -1), k=top_k*2)
        
        # Combine scores
        combined_scores = {}
        for uid, dist in zip(img_ids[0], img_dists[0]):
            combined_scores[uid] = image_weight * (1.0 / (1.0 + dist))
        
        for uid, dist in zip(txt_ids[0], txt_dists[0]):
            score = text_weight * (1.0 / (1.0 + dist))
            if uid in combined_scores:
                combined_scores[uid] += score
            else:
                combined_scores[uid] = score
        
        # Sort and get top-k
        sorted_candidates = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_uids = [uid for uid, _ in sorted_candidates[:top_k]]
        
        # Get results
        results = self.database.get_many(top_uids)
        
        # Add scores
        for result in results:
            result['score'] = combined_scores[result['uid']]
        
        elapsed = time.time() - start_time
        logger.info(f"Multimodal search completed in {elapsed*1000:.2f}ms")
        
        return results
    
    def print_results(self, results: List[Dict], max_text_length: int = 200):
        """Pretty print search results"""
        print("\n" + "="*80)
        print(f"Found {len(results)} results")
        print("="*80)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. UID: {result['uid']} | Score: {result['score']:.4f}")
            print(f"   Filename: {result['filename']}")
            print(f"   Projection: {result.get('projection', 'N/A')}")
            
            if result.get('mesh'):
                print(f"   MeSH: {', '.join(result['mesh'][:5])}")
            
            if result.get('problems'):
                print(f"   Problems: {', '.join(result['problems'][:5])}")
            
            findings = result.get('findings', '')
            if findings:
                truncated = findings[:max_text_length] + "..." if len(findings) > max_text_length else findings
                print(f"   Findings: {truncated}")
            
            impression = result.get('impression', '')
            if impression:
                truncated = impression[:max_text_length] + "..." if len(impression) > max_text_length else impression
                print(f"   Impression: {truncated}")
        
        print("\n" + "="*80 + "\n")


def demo():
    """Run demo of the retrieval system"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║   Medical Multimodal Retrieval System - Demo             ║
    ║   Indiana University Chest X-ray Dataset                 ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Initialize system
    system = MedicalRetrievalSystem(
        index_dir=INDEX_DIR,
        use_gpu=False  # Set to True if GPU available
    )
    
    # Demo 1: Text search
    print("\n📝 Demo 1: Search by Text Query")
    print("-" * 80)
    query_text = "pneumonia in right lower lobe"
    print(f"Query: '{query_text}'")
    
    results = system.search_by_text(
        query_text=query_text,
        top_k=5,
        enable_reranking=True
    )
    system.print_results(results)
    
    # Demo 2: Another text search
    print("\n📝 Demo 2: Search for Different Pathology")
    print("-" * 80)
    query_text = "pleural effusion left side"
    print(f"Query: '{query_text}'")
    
    results = system.search_by_text(
        query_text=query_text,
        top_k=5,
        enable_reranking=True
    )
    system.print_results(results)
    
    # Demo 3: MeSH term search
    print("\n📝 Demo 3: Search by Medical Terms")
    print("-" * 80)
    query_text = "cardiomegaly enlarged heart"
    print(f"Query: '{query_text}'")
    
    results = system.search_by_text(
        query_text=query_text,
        top_k=5,
        enable_reranking=True
    )
    system.print_results(results)
    
    print("\n✓ Demo completed successfully!")
    print("\nTo search with your own queries:")
    print("1. Use system.search_by_text(query_text) for text search")
    print("2. Use system.search_by_image(image) for image search")
    print("3. Use system.search_multimodal(text, image) for combined search")


if __name__ == "__main__":
    demo()
