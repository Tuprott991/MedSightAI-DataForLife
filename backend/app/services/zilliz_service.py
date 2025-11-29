"""
Zilliz Cloud Vector Database Service
Handles embedding storage and similarity search using Zilliz Cloud REST API
"""
import requests
from typing import List, Tuple, Optional, Dict, Any
from uuid import UUID
import hashlib
from fastapi import HTTPException
from app.config.settings import settings


class ZillizService:
    """Service for Zilliz Cloud operations using REST API"""
    
    def __init__(self):
        self.base_url = settings.ZILLIZ_CLOUD_URI
        self.api_key = settings.ZILLIZ_CLOUD_API_KEY
        self.collection_name = settings.ZILLIZ_COLLECTION_NAME
        self.txt_dimension = settings.ZILLIZ_TXT_DIMENSION
        self.img_dimension = settings.ZILLIZ_IMG_DIMENSION
        
        # Request headers
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def _uuid_to_int(self, uuid_str: str) -> int:
        """
        Convert UUID string to consistent integer for Zilliz primary_key
        Uses hash function to ensure consistent conversion
        """
        # Use first 15 digits of hash to fit in int64 range
        hash_val = int(hashlib.sha256(uuid_str.encode()).hexdigest(), 16)
        return hash_val % (2**63 - 1)  # Keep within int64 range
    
    def insert_embedding(
        self,
        case_id: str,
        txt_embedding: Optional[List[float]] = None,
        img_embedding: Optional[List[float]] = None
    ) -> bool:
        """
        Insert case embeddings into Zilliz
        
        Args:
            case_id: Case UUID as string
            txt_embedding: Text embedding vector (1152 dimensions)
            img_embedding: Image embedding vector (1152 dimensions)
        
        Returns:
            True if successful
        """
        try:
            primary_key = self._uuid_to_int(case_id)
            
            # Prepare data
            data = {"primary_key": primary_key}
            
            if txt_embedding:
                if len(txt_embedding) != self.txt_dimension:
                    raise ValueError(f"Text embedding must be {self.txt_dimension} dimensions")
                data["txt_emb"] = txt_embedding
            
            if img_embedding:
                if len(img_embedding) != self.img_dimension:
                    raise ValueError(f"Image embedding must be {self.img_dimension} dimensions")
                data["img_emb"] = img_embedding
            
            # Make request
            url = f"{self.base_url}/v2/vectordb/entities/insert"
            payload = {
                "collectionName": self.collection_name,
                "data": [data]
            }
            
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            
            result = response.json()
            if result.get("code") == 0:
                return True
            else:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Zilliz insert failed: {result.get('message', 'Unknown error')}"
                )
        
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Failed to insert embedding: {str(e)}")
    
    def upsert_embedding(
        self,
        case_id: str,
        txt_embedding: Optional[List[float]] = None,
        img_embedding: Optional[List[float]] = None
    ) -> bool:
        """
        Upsert (insert or update) case embeddings in Zilliz
        
        Args:
            case_id: Case UUID as string
            txt_embedding: Text embedding vector
            img_embedding: Image embedding vector
        
        Returns:
            True if successful
        """
        try:
            primary_key = self._uuid_to_int(case_id)
            
            # Prepare data
            data = {"primary_key": primary_key}
            
            if txt_embedding:
                data["txt_emb"] = txt_embedding
            
            if img_embedding:
                data["img_emb"] = img_embedding
            
            # Make request
            url = f"{self.base_url}/v2/vectordb/entities/upsert"
            payload = {
                "collectionName": self.collection_name,
                "data": [data]
            }
            
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            
            result = response.json()
            return result.get("code") == 0
        
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Failed to upsert embedding: {str(e)}")
    
    def search_similar_by_text(
        self,
        txt_embedding: List[float],
        top_k: int = 5
    ) -> Tuple[List[int], List[float]]:
        """
        Search for similar cases by text embedding
        
        Args:
            txt_embedding: Query text embedding vector
            top_k: Number of similar cases to return
        
        Returns:
            Tuple of (primary_keys, similarity_scores)
        """
        try:
            if len(txt_embedding) != self.txt_dimension:
                raise ValueError(f"Text embedding must be {self.txt_dimension} dimensions")
            
            url = f"{self.base_url}/v2/vectordb/entities/search"
            payload = {
                "collectionName": self.collection_name,
                "data": [txt_embedding],
                "annsField": "txt_emb",
                "limit": top_k,
                "outputFields": ["*"]
            }
            
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("code") != 0:
                raise HTTPException(
                    status_code=500,
                    detail=f"Search failed: {result.get('message', 'Unknown error')}"
                )
            
            # Extract results
            data = result.get("data", [])
            if isinstance(data, list) and len(data) > 0:
                search_results = data[0] if isinstance(data[0], list) else data
            else:
                search_results = []
            
            primary_keys = []
            scores = []
            for item in search_results:
                if isinstance(item, dict):
                    primary_keys.append(item.get("id") or item.get("primary_key"))
                    scores.append(item.get("distance", 0))
            
            return primary_keys, scores
        
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Failed to search similar cases: {str(e)}")
    
    def search_similar_by_image(
        self,
        img_embedding: List[float],
        top_k: int = 5
    ) -> Tuple[List[int], List[float]]:
        """
        Search for similar cases by image embedding
        
        Args:
            img_embedding: Query image embedding vector
            top_k: Number of similar cases to return
        
        Returns:
            Tuple of (primary_keys, similarity_scores)
        """
        try:
            if len(img_embedding) != self.img_dimension:
                raise ValueError(f"Image embedding must be {self.img_dimension} dimensions")
            
            url = f"{self.base_url}/v2/vectordb/entities/search"
            payload = {
                "collectionName": self.collection_name,
                "data": [img_embedding],
                "annsField": "img_emb",
                "limit": top_k,
                "outputFields": ["*"]
            }
            
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("code") != 0:
                raise HTTPException(
                    status_code=500,
                    detail=f"Search failed: {result.get('message', 'Unknown error')}"
                )
            
            # Extract results
            data = result.get("data", [])
            if isinstance(data, list) and len(data) > 0:
                search_results = data[0] if isinstance(data[0], list) else data
            else:
                search_results = []
            
            primary_keys = []
            scores = []
            for item in search_results:
                if isinstance(item, dict):
                    primary_keys.append(item.get("id") or item.get("primary_key"))
                    scores.append(item.get("distance", 0))
            
            return primary_keys, scores
        
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Failed to search similar cases: {str(e)}")
    
    def hybrid_search(
        self,
        txt_embedding: Optional[List[float]] = None,
        img_embedding: Optional[List[float]] = None,
        top_k: int = 5,
        rerank_strategy: str = "rrf",
        rerank_k: int = 10
    ) -> Tuple[List[int], List[float]]:
        """
        Hybrid search using both text and image embeddings
        
        Args:
            txt_embedding: Text embedding vector
            img_embedding: Image embedding vector
            top_k: Number of results to return
            rerank_strategy: Reranking strategy ("rrf" or "weighted")
            rerank_k: Reranking parameter
        
        Returns:
            Tuple of (primary_keys, similarity_scores)
        """
        try:
            search_requests = []
            
            if txt_embedding:
                if len(txt_embedding) != self.txt_dimension:
                    raise ValueError(f"Text embedding must be {self.txt_dimension} dimensions")
                search_requests.append({
                    "data": [txt_embedding],
                    "annsField": "txt_emb",
                    "limit": top_k,
                    "outputFields": ["*"]
                })
            
            if img_embedding:
                if len(img_embedding) != self.img_dimension:
                    raise ValueError(f"Image embedding must be {self.img_dimension} dimensions")
                search_requests.append({
                    "data": [img_embedding],
                    "annsField": "img_emb",
                    "limit": top_k,
                    "outputFields": ["*"]
                })
            
            if not search_requests:
                raise ValueError("Must provide at least one embedding (text or image)")
            
            url = f"{self.base_url}/v2/vectordb/entities/hybrid_search"
            payload = {
                "collectionName": self.collection_name,
                "search": search_requests,
                "rerank": {
                    "strategy": rerank_strategy,
                    "params": {"k": rerank_k}
                }
            }
            
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("code") != 0:
                raise HTTPException(
                    status_code=500,
                    detail=f"Hybrid search failed: {result.get('message', 'Unknown error')}"
                )
            
            # Extract results
            data = result.get("data", [])
            if isinstance(data, list) and len(data) > 0:
                # data is a list of lists, get first element
                search_results = data[0] if isinstance(data[0], list) else data
            else:
                search_results = []
            
            primary_keys = []
            scores = []
            
            for item in search_results:
                if isinstance(item, dict):
                    primary_keys.append(item.get("id") or item.get("primary_key"))
                    scores.append(item.get("distance", 0))
            
            return primary_keys, scores
        
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Failed to perform hybrid search: {str(e)}")
    
    def get_by_case_id(self, case_id: str) -> Optional[Dict[str, Any]]:
        """
        Get embedding by case ID
        
        Args:
            case_id: Case UUID as string
        
        Returns:
            Dictionary with embeddings or None if not found
        """
        try:
            primary_key = self._uuid_to_int(case_id)
            
            url = f"{self.base_url}/v2/vectordb/entities/get"
            payload = {
                "collectionName": self.collection_name,
                "id": [primary_key]
            }
            
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("code") != 0:
                return None
            
            data = result.get("data", [])
            return data[0] if data else None
        
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Failed to get embedding: {str(e)}")
    
    def query_by_filter(self, filter_expr: str) -> List[Dict[str, Any]]:
        """
        Query embeddings by filter expression
        
        Args:
            filter_expr: Filter expression (e.g., "primary_key in [1,2,3]")
        
        Returns:
            List of matching records
        """
        try:
            url = f"{self.base_url}/v2/vectordb/entities/query"
            payload = {
                "collectionName": self.collection_name,
                "filter": filter_expr
            }
            
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("code") != 0:
                raise HTTPException(
                    status_code=500,
                    detail=f"Query failed: {result.get('message', 'Unknown error')}"
                )
            
            return result.get("data", [])
        
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Failed to query: {str(e)}")
    
    def delete_by_case_id(self, case_id: str) -> bool:
        """
        Delete embedding by case ID
        
        Args:
            case_id: Case UUID as string
        
        Returns:
            True if successful
        """
        try:
            primary_key = self._uuid_to_int(case_id)
            
            url = f"{self.base_url}/v2/vectordb/entities/delete"
            payload = {
                "collectionName": self.collection_name,
                "filter": f"primary_key in [{primary_key}]"
            }
            
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            
            result = response.json()
            return result.get("code") == 0
        
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete embedding: {str(e)}")
    
    def delete_batch(self, case_ids: List[str]) -> bool:
        """
        Delete multiple embeddings by case IDs
        
        Args:
            case_ids: List of case UUID strings
        
        Returns:
            True if successful
        """
        try:
            primary_keys = [self._uuid_to_int(cid) for cid in case_ids]
            primary_keys_str = ",".join(map(str, primary_keys))
            
            url = f"{self.base_url}/v2/vectordb/entities/delete"
            payload = {
                "collectionName": self.collection_name,
                "filter": f"primary_key in [{primary_keys_str}]"
            }
            
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            
            result = response.json()
            return result.get("code") == 0
        
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete embeddings: {str(e)}")


# Create singleton instance
zilliz_service = ZillizService()
