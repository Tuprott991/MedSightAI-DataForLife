"""
Test script for image retrieval using Milvus vector database
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
import os

from milvus_setup import MedicalImageVectorDB


class ImageRetrieval:
    """
    Image retrieval system using Milvus vector database
    """
    
    def __init__(self, db: MedicalImageVectorDB):
        """
        Initialize retrieval system
        
        Args:
            db: Connected MedicalImageVectorDB instance
        """
        self.db = db
        
    def retrieve_similar_by_path(
        self,
        query_image_path: str,
        top_k: int = 10,
        mode: str = "image"
    ) -> List[Dict]:
        """
        Retrieve similar images by image path
        
        Args:
            query_image_path: Path to query image
            top_k: Number of similar images to return
            mode: "image" or "text" or "hybrid"
            
        Returns:
            List of similar images with metadata
        """
        # Get query image metadata from database
        filename = os.path.basename(query_image_path)
        
        # Query to find the image in database
        results = self.db.collection.query(
            expr=f'filename == "{filename}"',
            output_fields=["image_id", "filename"]
        )
        
        if not results:
            print(f"❌ Image not found in database: {filename}")
            return []
        
        # Get the image embedding from database
        image_id = results[0]['image_id']
        full_results = self.db.collection.query(
            expr=f'image_id == "{image_id}"',
            output_fields=["image_embedding", "text_embedding"]
        )
        
        image_embedding = np.array(full_results[0]['image_embedding'])
        
        # Search based on mode
        if mode == "image":
            similar = self.db.search_by_image(image_embedding, top_k=top_k)
        elif mode == "text":
            text_embedding = np.array(full_results[0]['text_embedding'])
            similar = self.db.search_by_text(text_embedding, top_k=top_k)
        elif mode == "hybrid":
            text_embedding = np.array(full_results[0]['text_embedding'])
            similar = self.db.search_hybrid(
                image_embedding,
                text_embedding,
                alpha=0.5,
                top_k=top_k
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        return similar
    
    def display_results(
        self,
        query_image_path: str,
        results: List[Dict],
        images_dir: str,
        max_display: int = 10
    ):
        """
        Display retrieval results with images
        
        Args:
            query_image_path: Path to query image
            results: Search results from Milvus
            images_dir: Directory containing images
            max_display: Maximum number of images to display
        """
        n_results = min(len(results), max_display)
        
        # Setup plot
        n_cols = 5
        n_rows = (n_results + n_cols) // n_cols + 1  # +1 for query row
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
        axes = axes.flatten()
        
        # Display query image
        try:
            query_img = Image.open(query_image_path)
            axes[0].imshow(query_img, cmap='gray')
            axes[0].set_title(
                f"QUERY IMAGE\n{os.path.basename(query_image_path)}",
                fontsize=12,
                fontweight='bold',
                color='red'
            )
            axes[0].axis('off')
        except Exception as e:
            print(f"Error loading query image: {e}")
        
        # Hide other cells in first row
        for i in range(1, n_cols):
            axes[i].axis('off')
        
        # Display retrieved images
        for i, result in enumerate(results[:n_results]):
            idx = n_cols + i
            
            # Get image path
            img_path = result['image_path']
            
            # If path is absolute kaggle path, replace with local path
            if '/kaggle/' in img_path:
                filename = result['filename']
                img_path = os.path.join(images_dir, filename)
            
            try:
                img = Image.open(img_path)
                axes[idx].imshow(img, cmap='gray')
                
                # Get similarity score
                similarity = result.get('similarity', result.get('combined_score', 0))
                
                axes[idx].set_title(
                    f"Rank {i+1}\nSimilarity: {similarity:.4f}\n{result['filename']}",
                    fontsize=10
                )
                axes[idx].axis('off')
            except Exception as e:
                axes[idx].text(
                    0.5, 0.5,
                    f"Error loading\n{result['filename']}",
                    ha='center',
                    va='center'
                )
                axes[idx].axis('off')
        
        # Hide remaining axes
        for i in range(n_cols + n_results, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed results
        print("\n" + "="*80)
        print("RETRIEVAL RESULTS")
        print("="*80)
        print(f"Query: {os.path.basename(query_image_path)}\n")
        
        for i, result in enumerate(results[:n_results]):
            similarity = result.get('similarity', result.get('combined_score', 0))
            print(f"Rank {i+1:2d} | Similarity: {similarity:.6f} | {result['filename']}")
            if 'report_text' in result:
                report_preview = result['report_text'][:100] + "..." if len(result['report_text']) > 100 else result['report_text']
                print(f"         Report: {report_preview}")
            print()


def main():
    """
    Main test function
    """
    print("="*80)
    print("Medical Image Retrieval Test with Milvus")
    print("="*80)
    
    # Initialize database
    db = MedicalImageVectorDB(
        host="localhost",
        port="19530",
        collection_name="medical_images_v1"
    )
    
    # Connect
    db.connect()
    
    # Load collection
    db.load_collection()
    
    # Get collection stats
    stats = db.get_collection_stats()
    print(f"\n📊 Collection Stats:")
    print(f"   Name: {stats['name']}")
    print(f"   Total images: {stats['num_entities']}")
    print(f"   Loaded: {stats['loaded']}")
    
    # Initialize retrieval system
    retrieval = ImageRetrieval(db)
    
    # Test query
    query_image_path = "/kaggle/input/chest-x-ray/images/images_normalized/1000_IM-0003-1001.dcm.png"
    images_dir = "/kaggle/input/chest-x-ray/images/images_normalized"
    
    print(f"\n🔍 Searching for similar images to: {os.path.basename(query_image_path)}")
    
    # Test different search modes
    for mode in ["image", "text", "hybrid"]:
        print(f"\n{'='*80}")
        print(f"Mode: {mode.upper()}")
        print('='*80)
        
        results = retrieval.retrieve_similar_by_path(
            query_image_path=query_image_path,
            top_k=10,
            mode=mode
        )
        
        if results:
            retrieval.display_results(
                query_image_path=query_image_path,
                results=results,
                images_dir=images_dir,
                max_display=10
            )
    
    # Disconnect
    db.disconnect()
    print("\n✅ Test complete!")


if __name__ == "__main__":
    main()
