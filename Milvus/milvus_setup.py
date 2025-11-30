"""
Milvus Vector Database Setup for MedSigLIP Embeddings
Optimized for storing and retrieving medical image and text embeddings
"""

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import json


class MedicalImageVectorDB:
    """
    Vector database for medical image and text embeddings using Milvus
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        collection_name: str = "medical_images"
    ):
        """
        Initialize Milvus connection
        
        Args:
            host: Milvus server host
            port: Milvus server port
            collection_name: Name of the collection
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.collection = None
        
    def connect(self):
        """Connect to Milvus server"""
        print(f"🔌 Connecting to Milvus at {self.host}:{self.port}...")
        connections.connect("default", host=self.host, port=self.port)
        print("✅ Connected to Milvus successfully!")
        
    def disconnect(self):
        """Disconnect from Milvus server"""
        connections.disconnect("default")
        print("🔌 Disconnected from Milvus")
        
    def create_collection(
        self,
        image_dim: int = 1152,
        text_dim: int = 1152,
        drop_existing: bool = False
    ):
        """
        Create a collection with optimized schema for medical images
        
        Args:
            image_dim: Dimension of image embeddings
            text_dim: Dimension of text embeddings
            drop_existing: Drop collection if it exists
        """
        # Drop existing collection if requested
        if drop_existing and utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print(f"🗑️ Dropped existing collection: {self.collection_name}")
        
        # Check if collection already exists
        if utility.has_collection(self.collection_name):
            print(f"✅ Collection '{self.collection_name}' already exists")
            self.collection = Collection(self.collection_name)
            return
        
        # Define schema
        fields = [
            # Primary key - auto-generated ID
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True
            ),
            
            # Image metadata
            FieldSchema(
                name="image_id",
                dtype=DataType.VARCHAR,
                max_length=256
            ),
            FieldSchema(
                name="filename",
                dtype=DataType.VARCHAR,
                max_length=512
            ),
            FieldSchema(
                name="image_path",
                dtype=DataType.VARCHAR,
                max_length=1024
            ),
            
            # Patient/Report metadata
            FieldSchema(
                name="uid",
                dtype=DataType.VARCHAR,
                max_length=256
            ),
            FieldSchema(
                name="report_text",
                dtype=DataType.VARCHAR,
                max_length=65535  # Max text length
            ),
            
            # Image embedding vector
            FieldSchema(
                name="image_embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=image_dim
            ),
            
            # Text embedding vector
            FieldSchema(
                name="text_embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=text_dim
            ),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Medical Image and Text Embeddings from MedSigLIP"
        )
        
        # Create collection
        print(f"📦 Creating collection: {self.collection_name}")
        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
            using='default',
            shards_num=2  # For better performance
        )
        print(f"✅ Collection '{self.collection_name}' created successfully!")
        
    def create_indexes(self):
        """
        Create optimized indexes for both image and text embeddings
        Uses IVF_FLAT for good balance between speed and accuracy
        """
        print("🔍 Creating indexes...")
        
        # Index parameters for IVF_FLAT
        # nlist: number of cluster units (sqrt of total entities is a good rule of thumb)
        index_params = {
            "metric_type": "IP",  # Inner Product (cosine similarity for normalized vectors)
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}  # Adjust based on dataset size
        }
        
        # Create index for image embeddings
        print("  📸 Creating index for image_embedding...")
        self.collection.create_index(
            field_name="image_embedding",
            index_params=index_params
        )
        
        # Create index for text embeddings
        print("  📝 Creating index for text_embedding...")
        self.collection.create_index(
            field_name="text_embedding",
            index_params=index_params
        )
        
        print("✅ Indexes created successfully!")
        
    def insert_data(
        self,
        image_ids: List[str],
        filenames: List[str],
        image_paths: List[str],
        uids: List[str],
        report_texts: List[str],
        image_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        batch_size: int = 1000
    ):
        """
        Insert data into Milvus collection in batches
        
        Args:
            image_ids: List of image IDs
            filenames: List of filenames
            image_paths: List of image paths
            uids: List of patient UIDs
            report_texts: List of report texts
            image_embeddings: Image embedding vectors (N, dim)
            text_embeddings: Text embedding vectors (N, dim)
            batch_size: Batch size for insertion
        """
        print(f"📥 Inserting {len(image_ids)} records into Milvus...")
        
        total = len(image_ids)
        for i in range(0, total, batch_size):
            end_idx = min(i + batch_size, total)
            
            batch_data = [
                image_ids[i:end_idx],
                filenames[i:end_idx],
                image_paths[i:end_idx],
                uids[i:end_idx],
                report_texts[i:end_idx],
                image_embeddings[i:end_idx].tolist(),
                text_embeddings[i:end_idx].tolist(),
            ]
            
            self.collection.insert(batch_data)
            print(f"  ✓ Inserted batch {i//batch_size + 1}/{(total-1)//batch_size + 1} ({end_idx}/{total})")
        
        # Flush to persist data
        self.collection.flush()
        print(f"✅ All {total} records inserted and flushed!")
        
    def load_collection(self):
        """Load collection into memory for searching"""
        print("⏳ Loading collection into memory...")
        self.collection.load()
        print("✅ Collection loaded!")
        
    def search_by_image(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        output_fields: List[str] = None
    ) -> List[Dict]:
        """
        Search similar images by image embedding
        
        Args:
            query_embedding: Query image embedding vector
            top_k: Number of top results to return
            output_fields: Fields to return in results
            
        Returns:
            List of search results with metadata
        """
        if output_fields is None:
            output_fields = ["image_id", "filename", "image_path", "uid", "report_text"]
        
        # Normalize query if needed
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 16}  # Search in 16 clusters
        }
        
        results = self.collection.search(
            data=[query_norm.tolist()],
            anns_field="image_embedding",
            param=search_params,
            limit=top_k,
            output_fields=output_fields
        )
        
        return self._format_results(results)
    
    def search_by_text(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        output_fields: List[str] = None
    ) -> List[Dict]:
        """
        Search similar images by text embedding
        
        Args:
            query_embedding: Query text embedding vector
            top_k: Number of top results to return
            output_fields: Fields to return in results
            
        Returns:
            List of search results with metadata
        """
        if output_fields is None:
            output_fields = ["image_id", "filename", "image_path", "uid", "report_text"]
        
        # Normalize query if needed
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 16}
        }
        
        results = self.collection.search(
            data=[query_norm.tolist()],
            anns_field="text_embedding",
            param=search_params,
            limit=top_k,
            output_fields=output_fields
        )
        
        return self._format_results(results)
    
    def search_hybrid(
        self,
        image_query: np.ndarray,
        text_query: np.ndarray,
        alpha: float = 0.5,
        top_k: int = 10,
        output_fields: List[str] = None
    ) -> List[Dict]:
        """
        Hybrid search combining image and text embeddings
        
        Args:
            image_query: Image embedding query
            text_query: Text embedding query
            alpha: Weight for image vs text (0-1, 0=text only, 1=image only)
            top_k: Number of results
            output_fields: Fields to return
            
        Returns:
            Combined search results
        """
        # Get results from both searches
        image_results = self.search_by_image(image_query, top_k=top_k*2, output_fields=output_fields)
        text_results = self.search_by_text(text_query, top_k=top_k*2, output_fields=output_fields)
        
        # Combine scores with alpha weighting
        combined_scores = {}
        for result in image_results:
            img_id = result['image_id']
            combined_scores[img_id] = {
                'score': alpha * result['similarity'],
                'data': result
            }
        
        for result in text_results:
            img_id = result['image_id']
            if img_id in combined_scores:
                combined_scores[img_id]['score'] += (1 - alpha) * result['similarity']
            else:
                combined_scores[img_id] = {
                    'score': (1 - alpha) * result['similarity'],
                    'data': result
                }
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )[:top_k]
        
        # Format output
        return [
            {**item[1]['data'], 'combined_score': item[1]['score']}
            for item in sorted_results
        ]
    
    def _format_results(self, results) -> List[Dict]:
        """Format Milvus search results into dictionary list"""
        formatted = []
        for hits in results:
            for hit in hits:
                result = {
                    'id': hit.id,
                    'similarity': hit.distance,
                }
                # Add all entity fields
                for field_name in hit.entity._row_data:
                    result[field_name] = hit.entity.get(field_name)
                formatted.append(result)
        return formatted
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        stats = {
            'name': self.collection_name,
            'num_entities': self.collection.num_entities,
            'loaded': utility.load_state(self.collection_name),
        }
        return stats
    
    def query_by_id(self, image_id: str) -> Dict:
        """Query record by image_id"""
        results = self.collection.query(
            expr=f'image_id == "{image_id}"',
            output_fields=["image_id", "filename", "image_path", "uid", "report_text"]
        )
        return results[0] if results else None


def load_and_insert_embeddings(
    db: MedicalImageVectorDB,
    merged_df_path: str,
    image_embeddings_path: str,
    text_embeddings_path: str
):
    """
    Load embeddings from files and insert into Milvus
    
    Args:
        db: MedicalImageVectorDB instance
        merged_df_path: Path to merged dataframe CSV
        image_embeddings_path: Path to image embeddings .npy
        text_embeddings_path: Path to text embeddings .npy
    """
    print("\n📂 Loading data from files...")
    
    # Load CSV
    df = pd.read_csv(merged_df_path)
    print(f"  ✓ Loaded {len(df)} records from CSV")
    
    # Load embeddings
    image_embeddings = np.load(image_embeddings_path)
    text_embeddings = np.load(text_embeddings_path)
    print(f"  ✓ Loaded image embeddings: {image_embeddings.shape}")
    print(f"  ✓ Loaded text embeddings: {text_embeddings.shape}")
    
    # Prepare data
    image_ids = df['uid'].astype(str).tolist()
    filenames = df['filename'].astype(str).tolist()
    image_paths = df['image_path'].astype(str).tolist()
    uids = df['uid'].astype(str).tolist()
    report_texts = df['report'].astype(str).tolist()
    
    # Normalize embeddings for cosine similarity (IP metric)
    print("\n🔄 Normalizing embeddings for cosine similarity...")
    image_embeddings_norm = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    text_embeddings_norm = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    
    # Insert into Milvus
    db.insert_data(
        image_ids=image_ids,
        filenames=filenames,
        image_paths=image_paths,
        uids=uids,
        report_texts=report_texts,
        image_embeddings=image_embeddings_norm,
        text_embeddings=text_embeddings_norm,
        batch_size=1000
    )


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("Medical Image Vector Database with Milvus")
    print("=" * 80)
    
    # Initialize database
    db = MedicalImageVectorDB(
        host="localhost",
        port="19530",
        collection_name="medical_images_v1"
    )
    
    # Connect
    db.connect()
    
    # Create collection with schema
    db.create_collection(
        image_dim=1152,
        text_dim=1152,
        drop_existing=True  # Set to False to keep existing data
    )
    
    # Create indexes
    db.create_indexes()
    
    # Load and insert data (uncomment when ready)
    # load_and_insert_embeddings(
    #     db=db,
    #     merged_df_path="merged_df.csv",
    #     image_embeddings_path="medsiglip_image_embeddings.npy",
    #     text_embeddings_path="medsiglip_text_embeddings.npy"
    # )
    
    # Load collection into memory
    # db.load_collection()
    
    # Get stats
    # stats = db.get_collection_stats()
    # print(f"\n📊 Collection Stats: {json.dumps(stats, indent=2)}")
    
    # Disconnect
    db.disconnect()
    
    print("\n✅ Setup complete!")
