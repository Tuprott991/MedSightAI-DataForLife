"""
Dataset loader for Indiana University Chest X-ray dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from PIL import Image

logger = logging.getLogger(__name__)


class IndianaDatasetLoader:
    """Load and parse Indiana University Chest X-ray dataset"""
    
    def __init__(self, 
                 reports_csv: str,
                 projections_csv: str,
                 images_dir: str):
        """
        Args:
            reports_csv: Path to Indiana_reports.csv
            projections_csv: Path to Indiana_projections.csv
            images_dir: Directory containing X-ray images
        """
        self.reports_csv = Path(reports_csv)
        self.projections_csv = Path(projections_csv)
        self.images_dir = Path(images_dir)
        
        self.reports_df = None
        self.projections_df = None
        self.merged_df = None
        
    def load(self) -> pd.DataFrame:
        """Load and merge all dataset files"""
        logger.info("Loading Indiana dataset...")
        
        # Load reports
        self.reports_df = pd.read_csv(self.reports_csv)
        logger.info(f"Loaded {len(self.reports_df)} reports")
        
        # Load projections
        self.projections_df = pd.read_csv(self.projections_csv)
        logger.info(f"Loaded {len(self.projections_df)} projections")
        
        # Merge on uid
        self.merged_df = self._merge_data()
        logger.info(f"Merged dataset: {len(self.merged_df)} records")
        
        return self.merged_df
    
    def _merge_data(self) -> pd.DataFrame:
        """Merge reports and projections data"""
        # Merge on uid column
        merged = pd.merge(
            self.projections_df,
            self.reports_df,
            on='uid',
            how='inner'
        )
        
        # Filter out records without images
        merged = merged[merged['filename'].notna()]
        
        # Add full image paths
        merged['image_path'] = merged['filename'].apply(
            lambda x: str(self.images_dir / x) if pd.notna(x) else None
        )
        
        # Verify images exist
        merged['image_exists'] = merged['image_path'].apply(
            lambda x: Path(x).exists() if x else False
        )
        
        valid_images = merged['image_exists'].sum()
        logger.info(f"Found {valid_images}/{len(merged)} valid images")
        
        # Keep only records with valid images
        merged = merged[merged['image_exists']].copy()
        
        return merged
    
    def get_sample(self, idx: int) -> Dict:
        """Get a single sample by index"""
        if self.merged_df is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        row = self.merged_df.iloc[idx]
        return self._row_to_dict(row)
    
    def _row_to_dict(self, row) -> Dict:
        """Convert DataFrame row to dictionary"""
        return {
            'uid': row.get('uid', ''),
            'image_path': row.get('image_path', ''),
            'filename': row.get('filename', ''),
            'projection': row.get('projection', ''),
            
            # Clinical information
            'mesh': self._parse_list_field(row.get('MeSH', '')),
            'problems': self._parse_list_field(row.get('Problems', '')),
            'findings': str(row.get('findings', '')),
            'impression': str(row.get('impression', '')),
            'indication': str(row.get('indication', '')),
            'comparison': str(row.get('comparison', '')),
        }
    
    def _parse_list_field(self, field: str) -> List[str]:
        """Parse comma-separated or semicolon-separated list fields"""
        if pd.isna(field) or field == '':
            return []
        
        # Handle multiple separators
        if ';' in field:
            items = field.split(';')
        elif ',' in field:
            items = field.split(',')
        else:
            items = [field]
        
        # Clean and filter
        items = [item.strip() for item in items if item.strip()]
        return items
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        if self.merged_df is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        stats = {
            'total_samples': len(self.merged_df),
            'unique_patients': self.merged_df['uid'].nunique(),
            'projections': self.merged_df['projection'].value_counts().to_dict(),
            'avg_findings_length': self.merged_df['findings'].str.len().mean(),
            'avg_impression_length': self.merged_df['impression'].str.len().mean(),
            'samples_with_mesh': self.merged_df['MeSH'].notna().sum(),
            'samples_with_problems': self.merged_df['Problems'].notna().sum(),
        }
        
        return stats
    
    def iterate_samples(self):
        """Iterate over all samples"""
        if self.merged_df is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        for idx, row in self.merged_df.iterrows():
            yield self._row_to_dict(row)
    
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load and return PIL Image"""
        try:
            img = Image.open(image_path).convert('RGB')
            return img
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None


def test_loader():
    """Test the dataset loader"""
    # Adjust these paths to your actual dataset location
    loader = IndianaDatasetLoader(
        reports_csv="data/indiana/Indiana_reports.csv",
        projections_csv="data/indiana/Indiana_projections.csv",
        images_dir="data/indiana/images"
    )
    
    # Load data
    df = loader.load()
    
    # Print statistics
    stats = loader.get_statistics()
    print("\n=== Dataset Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Print sample
    print("\n=== Sample Record ===")
    sample = loader.get_sample(0)
    for key, value in sample.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"{key}: {value[:100]}...")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_loader()
