"""
Reranking module for medical image retrieval
Combines multiple signals for improved ranking
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class MedicalReranker:
    """
    Rerank retrieved results using multiple clinical signals
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize reranker with custom weights
        
        Args:
            weights: Dictionary of feature weights
                - visual_similarity: Image embedding similarity
                - findings_similarity: Findings text match
                - impression_similarity: Impression text match
                - mesh_overlap: MeSH terms overlap
                - problems_overlap: Problems list overlap
        """
        # Default weights optimized for Indiana dataset
        self.default_weights = {
            "visual_similarity": 0.30,
            "findings_similarity": 0.25,
            "impression_similarity": 0.20,
            "mesh_overlap": 0.15,
            "problems_overlap": 0.10
        }
        
        self.weights = weights if weights is not None else self.default_weights
        
        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
        
        logger.info(f"Reranker initialized with weights: {self.weights}")
    
    def rerank(self,
               query: Dict,
               candidates: List[Dict],
               query_embeddings: Dict[str, np.ndarray],
               candidate_embeddings: List[Dict[str, np.ndarray]]) -> List[Tuple[Dict, float]]:
        """
        Rerank candidates based on multiple signals
        
        Args:
            query: Query dictionary with clinical information
            candidates: List of candidate dictionaries
            query_embeddings: Query embeddings (image, text)
            candidate_embeddings: List of candidate embeddings
            
        Returns:
            List of (candidate, score) tuples sorted by score
        """
        scores = []
        
        for candidate, cand_emb in zip(candidates, candidate_embeddings):
            score = self._compute_score(
                query, candidate,
                query_embeddings, cand_emb
            )
            scores.append((candidate, score))
        
        # Sort by score descending
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)
        
        return ranked
    
    def _compute_score(self,
                      query: Dict,
                      candidate: Dict,
                      query_emb: Dict[str, np.ndarray],
                      candidate_emb: Dict[str, np.ndarray]) -> float:
        """
        Compute overall score for a single candidate
        
        Args:
            query: Query information
            candidate: Candidate information
            query_emb: Query embeddings
            candidate_emb: Candidate embeddings
            
        Returns:
            Weighted score
        """
        # 1. Visual similarity
        visual_sim = self._visual_similarity(
            query_emb.get('image'),
            candidate_emb.get('image')
        )
        
        # 2. Findings similarity
        findings_sim = self._text_similarity(
            query_emb.get('findings'),
            candidate_emb.get('findings')
        )
        
        # 3. Impression similarity
        impression_sim = self._text_similarity(
            query_emb.get('impression'),
            candidate_emb.get('impression')
        )
        
        # 4. MeSH overlap
        mesh_overlap = self._compute_mesh_overlap(
            query.get('mesh', []),
            candidate.get('mesh', [])
        )
        
        # 5. Problems overlap
        problems_overlap = self._compute_problems_overlap(
            query.get('problems', []),
            candidate.get('problems', [])
        )
        
        # Weighted combination
        score = (
            self.weights['visual_similarity'] * visual_sim +
            self.weights['findings_similarity'] * findings_sim +
            self.weights['impression_similarity'] * impression_sim +
            self.weights['mesh_overlap'] * mesh_overlap +
            self.weights['problems_overlap'] * problems_overlap
        )
        
        return score
    
    def _visual_similarity(self, 
                          query_emb: Optional[np.ndarray],
                          candidate_emb: Optional[np.ndarray]) -> float:
        """Compute visual similarity"""
        if query_emb is None or candidate_emb is None:
            return 0.0
        
        # Cosine similarity
        sim = np.dot(query_emb.flatten(), candidate_emb.flatten())
        
        # Convert to [0, 1] range
        sim = (sim + 1) / 2
        
        return float(sim)
    
    def _text_similarity(self,
                        query_emb: Optional[np.ndarray],
                        candidate_emb: Optional[np.ndarray]) -> float:
        """Compute text embedding similarity"""
        if query_emb is None or candidate_emb is None:
            return 0.0
        
        # Cosine similarity
        sim = np.dot(query_emb.flatten(), candidate_emb.flatten())
        
        # Convert to [0, 1] range
        sim = (sim + 1) / 2
        
        return float(sim)
    
    def _compute_mesh_overlap(self,
                             query_mesh: List[str],
                             candidate_mesh: List[str]) -> float:
        """
        Compute MeSH terms overlap (Jaccard similarity)
        
        Args:
            query_mesh: Query MeSH terms
            candidate_mesh: Candidate MeSH terms
            
        Returns:
            Jaccard similarity [0, 1]
        """
        if not query_mesh or not candidate_mesh:
            return 0.0
        
        # Normalize terms
        query_set = set(term.lower().strip() for term in query_mesh)
        candidate_set = set(term.lower().strip() for term in candidate_mesh)
        
        # Jaccard similarity
        intersection = len(query_set & candidate_set)
        union = len(query_set | candidate_set)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _compute_problems_overlap(self,
                                  query_problems: List[str],
                                  candidate_problems: List[str]) -> float:
        """
        Compute problems list overlap (Jaccard similarity)
        
        Args:
            query_problems: Query problems
            candidate_problems: Candidate problems
            
        Returns:
            Jaccard similarity [0, 1]
        """
        if not query_problems or not candidate_problems:
            return 0.0
        
        # Normalize terms
        query_set = set(term.lower().strip() for term in query_problems)
        candidate_set = set(term.lower().strip() for term in candidate_problems)
        
        # Jaccard similarity
        intersection = len(query_set & candidate_set)
        union = len(query_set | candidate_set)
        
        if union == 0:
            return 0.0
        
        return intersection / union


class CrossEncoderReranker:
    """
    Advanced reranker using cross-encoder architecture
    More accurate but slower than feature-based reranking
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder reranker
        
        Args:
            model_name: Cross-encoder model name
        """
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            logger.info(f"Loaded cross-encoder: {model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")
            self.model = None
    
    def rerank(self,
               query_text: str,
               candidates: List[Dict]) -> List[Tuple[Dict, float]]:
        """
        Rerank using cross-encoder
        
        Args:
            query_text: Query text
            candidates: List of candidates with text fields
            
        Returns:
            Ranked list of (candidate, score)
        """
        if self.model is None:
            logger.warning("Cross-encoder not available, returning candidates unchanged")
            return [(c, 0.0) for c in candidates]
        
        # Prepare pairs
        pairs = []
        for candidate in candidates:
            # Combine candidate text fields
            candidate_text = self._combine_text(candidate)
            pairs.append([query_text, candidate_text])
        
        # Score pairs
        scores = self.model.predict(pairs)
        
        # Combine and sort
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return ranked
    
    def _combine_text(self, candidate: Dict) -> str:
        """Combine candidate text fields"""
        parts = []
        
        if candidate.get('findings'):
            parts.append(candidate['findings'])
        if candidate.get('impression'):
            parts.append(candidate['impression'])
        
        return " ".join(parts)


class HybridReranker:
    """
    Hybrid reranker combining multiple reranking strategies
    """
    
    def __init__(self,
                 feature_reranker: MedicalReranker,
                 cross_encoder_reranker: Optional[CrossEncoderReranker] = None,
                 alpha: float = 0.7):
        """
        Args:
            feature_reranker: Feature-based reranker
            cross_encoder_reranker: Cross-encoder reranker (optional)
            alpha: Weight for feature reranker (1-alpha for cross-encoder)
        """
        self.feature_reranker = feature_reranker
        self.cross_encoder_reranker = cross_encoder_reranker
        self.alpha = alpha
    
    def rerank(self,
               query: Dict,
               candidates: List[Dict],
               query_embeddings: Dict[str, np.ndarray],
               candidate_embeddings: List[Dict[str, np.ndarray]]) -> List[Tuple[Dict, float]]:
        """
        Hybrid reranking combining both methods
        """
        # Feature-based ranking
        feature_ranked = self.feature_reranker.rerank(
            query, candidates, query_embeddings, candidate_embeddings
        )
        
        if self.cross_encoder_reranker is None:
            return feature_ranked
        
        # Cross-encoder ranking
        query_text = query.get('findings', '') + ' ' + query.get('impression', '')
        cross_ranked = self.cross_encoder_reranker.rerank(query_text, candidates)
        
        # Combine scores
        # Create score dictionaries
        feature_scores = {id(cand): score for cand, score in feature_ranked}
        cross_scores = {id(cand): score for cand, score in cross_ranked}
        
        # Normalize scores to [0, 1]
        def normalize_scores(scores_dict):
            values = list(scores_dict.values())
            if not values:
                return scores_dict
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                return {k: 0.5 for k in scores_dict}
            return {k: (v - min_val) / (max_val - min_val) for k, v in scores_dict.items()}
        
        feature_scores = normalize_scores(feature_scores)
        cross_scores = normalize_scores(cross_scores)
        
        # Weighted combination
        final_scores = []
        for candidate in candidates:
            cand_id = id(candidate)
            combined_score = (
                self.alpha * feature_scores.get(cand_id, 0) +
                (1 - self.alpha) * cross_scores.get(cand_id, 0)
            )
            final_scores.append((candidate, combined_score))
        
        # Sort by combined score
        final_ranked = sorted(final_scores, key=lambda x: x[1], reverse=True)
        
        return final_ranked


def test_reranker():
    """Test the reranking module"""
    print("=== Testing Medical Reranker ===\n")
    
    # Create dummy data
    query = {
        'mesh': ['Pneumonia', 'Lung Diseases'],
        'problems': ['pneumonia'],
        'findings': 'Opacity in right lower lobe',
        'impression': 'Pneumonia'
    }
    
    candidates = [
        {
            'uid': '001',
            'mesh': ['Pneumonia'],
            'problems': ['pneumonia'],
            'findings': 'Right lower lobe infiltrate',
            'impression': 'Pneumonia'
        },
        {
            'uid': '002',
            'mesh': ['Pleural Effusion'],
            'problems': ['effusion'],
            'findings': 'Fluid in left hemithorax',
            'impression': 'Pleural effusion'
        }
    ]
    
    # Dummy embeddings
    query_embeddings = {
        'image': np.random.randn(512),
        'findings': np.random.randn(512),
        'impression': np.random.randn(512)
    }
    
    candidate_embeddings = [
        {
            'image': np.random.randn(512),
            'findings': np.random.randn(512),
            'impression': np.random.randn(512)
        }
        for _ in candidates
    ]
    
    # Test reranking
    reranker = MedicalReranker()
    ranked = reranker.rerank(query, candidates, query_embeddings, candidate_embeddings)
    
    print("Ranked results:")
    for i, (candidate, score) in enumerate(ranked, 1):
        print(f"{i}. UID: {candidate['uid']}, Score: {score:.4f}")
    
    print("\n✓ Reranker test passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_reranker()
