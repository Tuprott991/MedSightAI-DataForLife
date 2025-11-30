# Medical Image Vector Database with Milvus

Optimized vector database system for storing and retrieving medical image and text embeddings using Milvus.

## 🚀 Features

- **Dual Vector Search**: Search by image embeddings, text embeddings, or hybrid
- **Optimized Indexing**: IVF_FLAT index for fast similarity search
- **Scalable**: Handles millions of vectors efficiently
- **Metadata Support**: Store and retrieve image paths, filenames, reports, etc.
- **Batch Operations**: Efficient batch insertion for large datasets
- **Docker-based**: Easy setup with Docker Compose

## 📋 Prerequisites

- Docker & Docker Compose
- Python 3.8+
- NVIDIA GPU (optional, for faster processing)

## 🛠️ Installation

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Milvus with Docker Compose

```bash
docker-compose up -d
```

This will start:
- Milvus standalone server (port 19530)
- etcd (metadata storage)
- MinIO (object storage)

### 3. Verify Milvus is running

```bash
docker ps
```

You should see three containers running:
- `milvus-standalone`
- `milvus-etcd`
- `milvus-minio`

## 💾 Database Schema

### Collection: `medical_images_v1`

**Fields:**
- `id` (INT64, Primary Key): Auto-generated ID
- `image_id` (VARCHAR): Unique image identifier
- `filename` (VARCHAR): Image filename
- `image_path` (VARCHAR): Full path to image
- `uid` (VARCHAR): Patient/case UID
- `report_text` (VARCHAR): Medical report text
- `image_embedding` (FLOAT_VECTOR, dim=1152): Image embedding from MedSigLIP
- `text_embedding` (FLOAT_VECTOR, dim=1152): Text embedding from MedSigLIP

**Indexes:**
- `image_embedding`: IVF_FLAT index with IP (Inner Product) metric
- `text_embedding`: IVF_FLAT index with IP metric

## 📖 Usage

### 1. Setup Database and Insert Data

```python
from milvus_setup import MedicalImageVectorDB, load_and_insert_embeddings

# Initialize
db = MedicalImageVectorDB(
    host="localhost",
    port="19530",
    collection_name="medical_images_v1"
)

# Connect
db.connect()

# Create collection
db.create_collection(image_dim=1152, text_dim=1152, drop_existing=False)

# Create indexes
db.create_indexes()

# Load and insert embeddings
load_and_insert_embeddings(
    db=db,
    merged_df_path="merged_df.csv",
    image_embeddings_path="medsiglip_image_embeddings.npy",
    text_embeddings_path="medsiglip_text_embeddings.npy"
)

# Load into memory
db.load_collection()
```

### 2. Search Similar Images

```python
import numpy as np

# Load collection
db.load_collection()

# Search by image embedding
query_embedding = np.random.rand(1152)  # Replace with actual embedding
results = db.search_by_image(query_embedding, top_k=10)

for result in results:
    print(f"Similarity: {result['similarity']:.4f}")
    print(f"Filename: {result['filename']}")
    print(f"Report: {result['report_text'][:100]}...")
    print()
```

### 3. Hybrid Search (Image + Text)

```python
# Search with both image and text embeddings
results = db.search_hybrid(
    image_query=image_embedding,
    text_query=text_embedding,
    alpha=0.5,  # 0.5 = equal weight, 1.0 = image only, 0.0 = text only
    top_k=10
)
```

### 4. Run Retrieval Test

```python
python retrieval_test.py
```

## 🔧 Configuration

### Adjust Index Parameters

For better performance with your dataset size:

```python
# In milvus_setup.py, modify index_params:
index_params = {
    "metric_type": "IP",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}  # Increase for larger datasets
}
```

**Rule of thumb for `nlist`:**
- Small dataset (<100K): 128
- Medium dataset (100K-1M): 1024
- Large dataset (>1M): 4096

### Search Parameters

Adjust `nprobe` for speed vs accuracy trade-off:

```python
search_params = {
    "metric_type": "IP",
    "params": {"nprobe": 16}  # Higher = more accurate but slower
}
```

## 📊 Performance Optimization

### 1. Batch Size
- Insertion: 1000-5000 records per batch
- Adjust based on your memory

### 2. Index Type Selection
- `IVF_FLAT`: Best accuracy, moderate speed (recommended)
- `IVF_SQ8`: Faster, lower memory, slight accuracy loss
- `IVF_PQ`: Fastest, lowest memory, more accuracy loss
- `HNSW`: Best for small-medium datasets (<1M)

### 3. Memory Management
```python
# Increase Docker memory if needed
# Edit docker-compose.yml:
services:
  standalone:
    deploy:
      resources:
        limits:
          memory: 8G
```

## 🐛 Troubleshooting

### Connection Error
```bash
# Check if Milvus is running
docker ps | grep milvus

# Check logs
docker logs milvus-standalone
```

### Out of Memory
```bash
# Restart Milvus
docker-compose restart

# Or increase memory limit in docker-compose.yml
```

### Slow Search
- Reduce `nprobe` value
- Use IVF_SQ8 or IVF_PQ index
- Increase `nlist` during index creation

## 📚 API Reference

### MedicalImageVectorDB Class

**Methods:**
- `connect()`: Connect to Milvus
- `disconnect()`: Disconnect from Milvus
- `create_collection()`: Create collection with schema
- `create_indexes()`: Create vector indexes
- `insert_data()`: Batch insert embeddings
- `load_collection()`: Load collection into memory
- `search_by_image()`: Search by image embedding
- `search_by_text()`: Search by text embedding
- `search_hybrid()`: Hybrid search
- `get_collection_stats()`: Get collection statistics
- `query_by_id()`: Query specific record

## 🔗 Resources

- [Milvus Documentation](https://milvus.io/docs)
- [PyMilvus API](https://milvus.io/api-reference/pymilvus/v2.3.x/About.md)
- [MedSigLIP Model](https://huggingface.co/google/medsiglip-448)

## 📝 License

MIT License

## 🤝 Contributing

Feel free to submit issues and pull requests!
