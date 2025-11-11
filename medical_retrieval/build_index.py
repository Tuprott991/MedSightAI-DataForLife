"""
Build index from Indiana dataset
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm
import logging
from typing import Dict, List
import argparse

from data.dataset_loader import IndianaDatasetLoader
from data.preprocessor import TextPreprocessor
from models.encoder import MedCLIPEncoder
from indexing.faiss_index import FAISSIndex
from indexing.database import MetadataDatabase
from config import (
    MODEL_CONFIG, INDEX_CONFIG, INDIANA_CONFIG, 
    INDEX_DIR, DATA_DIR
)
from model_selector import get_active_model_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndexBuilder:
    """Build and populate FAISS index from Indiana dataset"""
    
    def __init__(self,
                 dataset_loader: IndianaDatasetLoader,
                 encoder: MedCLIPEncoder,
                 database: MetadataDatabase,
                 image_index: FAISSIndex,
                 text_index: FAISSIndex,
                 use_cache: bool = True):
        """
        Initialize index builder
        
        Args:
            dataset_loader: Indiana dataset loader
            encoder: MedCLIP encoder
            database: Metadata database
            image_index: FAISS index for images
            text_index: FAISS index for text
            use_cache: Whether to use cached embeddings
        """
        self.dataset_loader = dataset_loader
        self.encoder = encoder
        self.database = database
        self.image_index = image_index
        self.text_index = text_index
        self.use_cache = use_cache
        
        self.text_preprocessor = TextPreprocessor()
    
    def build(self, batch_size: int = 32):
        """
        Build indexes from dataset
        
        Args:
            batch_size: Batch size for encoding
        """
        logger.info("Starting index building...")
        
        # Load dataset
        df = self.dataset_loader.load()
        total_samples = len(df)
        
        logger.info(f"Building index for {total_samples} samples")
        
        # Process in batches
        image_embeddings_batch = []
        text_embeddings_batch = []
        metadata_batch = []
        uids_batch = []
        
        for idx, sample in enumerate(tqdm(self.dataset_loader.iterate_samples(), 
                                          total=total_samples,
                                          desc="Processing samples")):
            try:
                # Check cache
                if self.use_cache:
                    cached_img_emb = self.database.get_cached_embedding(sample['uid'], 'image')
                    cached_txt_emb = self.database.get_cached_embedding(sample['uid'], 'findings')
                    
                    if cached_img_emb is not None and cached_txt_emb is not None:
                        image_embeddings_batch.append(cached_img_emb)
                        text_embeddings_batch.append(cached_txt_emb)
                        uids_batch.append(sample['uid'])
                        metadata_batch.append(sample)
                        continue
                
                # Load and encode image
                image = self.dataset_loader.load_image(sample['image_path'])
                if image is None:
                    logger.warning(f"Failed to load image: {sample['image_path']}")
                    continue
                
                img_embedding = self.encoder.encode_image(image, normalize=True)[0]
                
                # Prepare and encode text
                combined_text = self.text_preprocessor.combine_clinical_text(
                    findings=sample['findings'],
                    impression=sample['impression'],
                    indication=sample['indication'],
                    use_template=True
                )
                
                if not combined_text:
                    combined_text = "No findings reported"
                
                txt_embedding = self.encoder.encode_text(combined_text, normalize=True)[0]
                
                # Add to batches
                image_embeddings_batch.append(img_embedding)
                text_embeddings_batch.append(txt_embedding)
                uids_batch.append(sample['uid'])
                metadata_batch.append(sample)
                
                # Cache embeddings
                if self.use_cache:
                    self.database.cache_embedding(sample['uid'], 'image', img_embedding)
                    self.database.cache_embedding(sample['uid'], 'findings', txt_embedding)
                
                # Process batch when full
                if len(image_embeddings_batch) >= batch_size:
                    self._add_batch_to_indexes(
                        image_embeddings_batch,
                        text_embeddings_batch,
                        uids_batch,
                        metadata_batch
                    )
                    
                    # Clear batches
                    image_embeddings_batch = []
                    text_embeddings_batch = []
                    uids_batch = []
                    metadata_batch = []
                
            except Exception as e:
                logger.error(f"Error processing sample {sample['uid']}: {e}")
                continue
        
        # Process remaining samples
        if image_embeddings_batch:
            self._add_batch_to_indexes(
                image_embeddings_batch,
                text_embeddings_batch,
                uids_batch,
                metadata_batch
            )
        
        logger.info("Index building completed!")
        logger.info(f"Image index: {self.image_index.index.ntotal} vectors")
        logger.info(f"Text index: {self.text_index.index.ntotal} vectors")
        logger.info(f"Database: {self.database.count()} records")
    
    def _add_batch_to_indexes(self,
                             image_embeddings: List[np.ndarray],
                             text_embeddings: List[np.ndarray],
                             uids: List[str],
                             metadata: List[Dict]):
        """Add batch to indexes and database"""
        # Convert to arrays
        img_emb_array = np.array(image_embeddings, dtype='float32')
        txt_emb_array = np.array(text_embeddings, dtype='float32')
        
        # Add to indexes
        self.image_index.add(img_emb_array, uids)
        self.text_index.add(txt_emb_array, uids)
        
        # Add to database
        self.database.insert_many(metadata)
    
    def save(self, index_dir: str):
        """Save indexes and database"""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving indexes to {index_dir}...")
        
        # Save FAISS indexes
        self.image_index.save(str(index_dir / "image_index.faiss"))
        self.text_index.save(str(index_dir / "text_index.faiss"))
        
        logger.info("Indexes saved successfully!")


def main():
    """Main function to build indexes"""
    parser = argparse.ArgumentParser(description="Build indexes for Indiana dataset")
    parser.add_argument("--reports-csv", type=str, required=True,
                       help="Path to Indiana_reports.csv")
    parser.add_argument("--projections-csv", type=str, required=True,
                       help="Path to Indiana_projections.csv")
    parser.add_argument("--images-dir", type=str, required=True,
                       help="Path to images directory")
    parser.add_argument("--output-dir", type=str, default=str(INDEX_DIR),
                       help="Output directory for indexes")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for encoding")
    parser.add_argument("--use-gpu", action="store_true",
                       help="Use GPU for encoding")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable embedding caching")
    
    args = parser.parse_args()
    
    # Initialize components
    logger.info("Initializing components...")
    
    # Dataset loader
    dataset_loader = IndianaDatasetLoader(
        reports_csv=args.reports_csv,
        projections_csv=args.projections_csv,
        images_dir=args.images_dir
    )
    
    # Encoder
    device = "cuda" if args.use_gpu else "cpu"
    model_config = get_active_model_config()
    
    logger.info(f"Using model: {model_config['active_model']}")
    logger.info(f"Model name: {model_config['model_name']}")
    
    encoder = MedCLIPEncoder(
        model_name=model_config['model_name'],
        model_type=model_config.get('model_type', 'clip'),
        device=device
    )
    
    # Database
    db_path = Path(args.output_dir) / "metadata.db"
    database = MetadataDatabase(str(db_path))
    
    # FAISS indexes
    embedding_dim = encoder.get_embedding_dim()
    
    image_index = FAISSIndex(
        dimension=embedding_dim,
        index_type=INDEX_CONFIG['type'],
        use_gpu=args.use_gpu,
        gpu_id=INDEX_CONFIG['gpu_id']
    )
    
    text_index = FAISSIndex(
        dimension=embedding_dim,
        index_type=INDEX_CONFIG['type'],
        use_gpu=args.use_gpu,
        gpu_id=INDEX_CONFIG['gpu_id']
    )
    
    # Index builder
    builder = IndexBuilder(
        dataset_loader=dataset_loader,
        encoder=encoder,
        database=database,
        image_index=image_index,
        text_index=text_index,
        use_cache=not args.no_cache
    )
    
    # Build indexes
    builder.build(batch_size=args.batch_size)
    
    # Save
    builder.save(args.output_dir)
    
    # Print statistics
    stats = database.get_statistics()
    logger.info("\n=== Final Statistics ===")
    for key, value in stats.items():
        logger.info(f"{key}: {value}")
    
    logger.info("\n✓ Index building completed successfully!")


if __name__ == "__main__":
    main()
