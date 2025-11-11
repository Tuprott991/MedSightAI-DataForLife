"""
Simple test client for API
"""
import requests
import json
from pathlib import Path
from typing import Dict, List

BASE_URL = "http://localhost:8000"


class MedicalRetrievalClient:
    """Client for Medical Retrieval API"""
    
    def __init__(self, base_url: str = BASE_URL):
        """Initialize client"""
        self.base_url = base_url
    
    def health_check(self) -> Dict:
        """Check API health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        response = requests.get(f"{self.base_url}/stats")
        return response.json()
    
    def search_by_text(self,
                      query_text: str,
                      top_k: int = 10,
                      enable_reranking: bool = True) -> Dict:
        """
        Search by text
        
        Args:
            query_text: Text query
            top_k: Number of results
            enable_reranking: Enable reranking
            
        Returns:
            Search response
        """
        payload = {
            "query_text": query_text,
            "top_k": top_k,
            "enable_reranking": enable_reranking
        }
        
        response = requests.post(
            f"{self.base_url}/search/text",
            json=payload
        )
        
        return response.json()
    
    def search_by_image(self,
                       image_path: str,
                       top_k: int = 10,
                       enable_reranking: bool = True) -> Dict:
        """
        Search by image
        
        Args:
            image_path: Path to image file
            top_k: Number of results
            enable_reranking: Enable reranking
            
        Returns:
            Search response
        """
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'top_k': top_k,
                'enable_reranking': enable_reranking
            }
            
            response = requests.post(
                f"{self.base_url}/search/image",
                files=files,
                data=data
            )
        
        return response.json()
    
    def search_multimodal(self,
                         image_path: str,
                         query_text: str,
                         top_k: int = 10,
                         image_weight: float = 0.5,
                         text_weight: float = 0.5) -> Dict:
        """
        Search by image and text
        
        Args:
            image_path: Path to image file
            query_text: Text query
            top_k: Number of results
            image_weight: Weight for image
            text_weight: Weight for text
            
        Returns:
            Search response
        """
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'query_text': query_text,
                'top_k': top_k,
                'image_weight': image_weight,
                'text_weight': text_weight
            }
            
            response = requests.post(
                f"{self.base_url}/search/multimodal",
                files=files,
                data=data
            )
        
        return response.json()
    
    def print_results(self, response: Dict):
        """Pretty print search results"""
        print("\n" + "="*80)
        print(f"Query: {response['query_info']}")
        print(f"Total Results: {response['total_results']}")
        print(f"Processing Time: {response['processing_time_ms']:.2f}ms")
        print("="*80)
        
        for i, result in enumerate(response['results'], 1):
            print(f"\n{i}. UID: {result['uid']} | Score: {result['score']:.4f}")
            print(f"   Filename: {result['filename']}")
            print(f"   Projection: {result['projection']}")
            
            if result['mesh']:
                print(f"   MeSH: {', '.join(result['mesh'][:3])}")
            
            if result['findings']:
                findings = result['findings'][:150] + "..." if len(result['findings']) > 150 else result['findings']
                print(f"   Findings: {findings}")
            
            if result['impression']:
                impression = result['impression'][:150] + "..." if len(result['impression']) > 150 else result['impression']
                print(f"   Impression: {impression}")
        
        print("\n" + "="*80 + "\n")


def test_api():
    """Test API endpoints"""
    print("🧪 Testing Medical Retrieval API\n")
    
    # Initialize client
    client = MedicalRetrievalClient()
    
    # 1. Health check
    print("1. Health Check...")
    health = client.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Total Images: {health['total_images']}")
    
    # 2. Statistics
    print("\n2. Database Statistics...")
    stats = client.get_stats()
    print(f"   Total Images: {stats['total_images']}")
    print(f"   Projections: {stats['by_projection']}")
    
    # 3. Text search
    print("\n3. Text Search...")
    response = client.search_by_text(
        query_text="pneumonia in right lower lobe",
        top_k=5,
        enable_reranking=True
    )
    client.print_results(response)
    
    # 4. More text searches
    test_queries = [
        "cardiomegaly",
        "pleural effusion left",
        "normal chest radiograph"
    ]
    
    for query in test_queries:
        print(f"\n4. Testing query: '{query}'")
        response = client.search_by_text(query, top_k=3)
        client.print_results(response)
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line usage
        client = MedicalRetrievalClient()
        
        if sys.argv[1] == "health":
            print(json.dumps(client.health_check(), indent=2))
        
        elif sys.argv[1] == "stats":
            print(json.dumps(client.get_stats(), indent=2))
        
        elif sys.argv[1] == "search" and len(sys.argv) > 2:
            query = " ".join(sys.argv[2:])
            response = client.search_by_text(query)
            client.print_results(response)
        
        else:
            print("Usage:")
            print("  python test_client.py health")
            print("  python test_client.py stats")
            print("  python test_client.py search <query>")
    
    else:
        # Run all tests
        test_api()
