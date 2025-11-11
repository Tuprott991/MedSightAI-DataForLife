"""
Image and text preprocessing for medical retrieval
"""
import torch
import numpy as np
from PIL import Image
from typing import Union, List, Optional
import cv2
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocess medical images for encoding"""
    
    def __init__(self, target_size: int = 224, normalize: bool = True):
        """
        Args:
            target_size: Target image size (default 224 for ViT)
            normalize: Whether to apply normalization
        """
        self.target_size = target_size
        self.normalize = normalize
        
        # ImageNet normalization (standard for CLIP models)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def preprocess(self, image: Union[Image.Image, np.ndarray, str]) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: PIL Image, numpy array, or image path
            
        Returns:
            Preprocessed image tensor [C, H, W]
        """
        # Load image if path
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Resize
        image = image.resize((self.target_size, self.target_size), Image.BILINEAR)
        
        # Convert to array
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Apply normalization
        if self.normalize:
            image_array = (image_array - self.mean) / self.std
        
        # Convert to tensor [C, H, W]
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        
        return image_tensor
    
    def preprocess_batch(self, images: List[Union[Image.Image, str]]) -> torch.Tensor:
        """
        Preprocess batch of images
        
        Args:
            images: List of PIL Images or image paths
            
        Returns:
            Batch tensor [B, C, H, W]
        """
        processed = [self.preprocess(img) for img in images]
        return torch.stack(processed)
    
    def apply_clahe(self, image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Useful for enhancing medical images
        
        Args:
            image: Input image (grayscale or RGB)
            clip_limit: CLAHE clip limit
            
        Returns:
            Enhanced image
        """
        if len(image.shape) == 3:
            # Convert RGB to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced


class TextPreprocessor:
    """Preprocess clinical text for encoding"""
    
    def __init__(self, max_length: int = 77):
        """
        Args:
            max_length: Maximum text length (default 77 for CLIP)
        """
        self.max_length = max_length
    
    def preprocess(self, text: str) -> str:
        """
        Clean and preprocess text
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text or text == 'nan' or text == '':
            return ""
        
        # Convert to string and strip
        text = str(text).strip()
        
        # Remove multiple spaces
        text = ' '.join(text.split())
        
        # Truncate if too long (preserve important info)
        if len(text) > self.max_length * 4:  # Rough estimate for tokens
            text = text[:self.max_length * 4]
        
        return text
    
    def combine_clinical_text(self, 
                             findings: str = "",
                             impression: str = "",
                             indication: str = "",
                             use_template: bool = True) -> str:
        """
        Combine multiple clinical text fields into one
        
        Args:
            findings: Findings text
            impression: Impression text
            indication: Indication text
            use_template: Whether to use structured template
            
        Returns:
            Combined text
        """
        findings = self.preprocess(findings)
        impression = self.preprocess(impression)
        indication = self.preprocess(indication)
        
        if use_template:
            # Structured template (better for medical LLMs)
            parts = []
            if indication:
                parts.append(f"Indication: {indication}")
            if findings:
                parts.append(f"Findings: {findings}")
            if impression:
                parts.append(f"Impression: {impression}")
            
            return ". ".join(parts)
        else:
            # Simple concatenation
            parts = [p for p in [indication, findings, impression] if p]
            return " ".join(parts)
    
    def extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """
        Extract medical keywords from text
        
        Args:
            text: Input text
            min_length: Minimum word length
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction (can be enhanced with medical NER)
        text = self.preprocess(text).lower()
        
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                    'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were'}
        
        words = text.split()
        keywords = [w for w in words if len(w) >= min_length and w not in stopwords]
        
        return keywords


class ClinicalFeatureExtractor:
    """Extract structured features from clinical data"""
    
    @staticmethod
    def extract_mesh_features(mesh_list: List[str]) -> dict:
        """
        Extract features from MeSH terms
        
        Args:
            mesh_list: List of MeSH terms
            
        Returns:
            Dictionary of MeSH features
        """
        return {
            'mesh_terms': mesh_list,
            'mesh_count': len(mesh_list),
            'has_mesh': len(mesh_list) > 0
        }
    
    @staticmethod
    def extract_problem_features(problems_list: List[str]) -> dict:
        """
        Extract features from problems list
        
        Args:
            problems_list: List of clinical problems
            
        Returns:
            Dictionary of problem features
        """
        return {
            'problems': problems_list,
            'problem_count': len(problems_list),
            'has_problems': len(problems_list) > 0
        }
    
    @staticmethod
    def normalize_mesh_term(term: str) -> str:
        """Normalize MeSH term for comparison"""
        return term.lower().strip().replace('/', '_')


def test_preprocessors():
    """Test preprocessing functions"""
    print("=== Testing Image Preprocessor ===")
    img_preprocessor = ImagePreprocessor()
    
    # Create dummy image
    dummy_img = Image.new('RGB', (512, 512), color='white')
    processed = img_preprocessor.preprocess(dummy_img)
    print(f"Processed image shape: {processed.shape}")
    
    print("\n=== Testing Text Preprocessor ===")
    text_preprocessor = TextPreprocessor()
    
    sample_text = """
    Findings: There is a focal opacity in the right lower lobe consistent with pneumonia.
    No pleural effusion or pneumothorax. Heart size is normal.
    """
    
    cleaned = text_preprocessor.preprocess(sample_text)
    print(f"Cleaned text: {cleaned}")
    
    keywords = text_preprocessor.extract_keywords(sample_text)
    print(f"Keywords: {keywords}")


if __name__ == "__main__":
    test_preprocessors()
