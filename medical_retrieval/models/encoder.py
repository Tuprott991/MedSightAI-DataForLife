"""
Medical Multimodal Encoder wrapper supporting CLIP and SigLIP architectures
"""
import torch
import torch.nn as nn
from transformers import (
    CLIPProcessor, CLIPModel, 
    AutoProcessor, AutoModel,
    SiglipProcessor, SiglipModel
)
from PIL import Image
from typing import Union, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MedCLIPEncoder:
    """
    Multimodal encoder supporting multiple architectures:
    - CLIP-based: PubMed CLIP, BiomedCLIP
    - SigLIP-based: SigLIP (Google), MedSigLIP (if available)
    """
    
    def __init__(self, 
                 model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                 model_type: str = "clip",
                 device: str = "cuda",
                 cache_dir: Optional[str] = None):
        """
        Initialize medical multimodal encoder
        
        Args:
            model_name: HuggingFace model name
                CLIP options:
                - "flaviagiammarino/pubmed-clip-vit-base-patch32" (PubMed CLIP)
                - "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224" (BiomedCLIP - RECOMMENDED)
                SigLIP options:
                - "google/siglip-base-patch16-224" (SigLIP base)
                - "google/siglip-large-patch16-384" (SigLIP large)
            model_type: Type of model ("clip" or "siglip")
            device: Device to run model on
            cache_dir: Cache directory for model weights
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model_type = model_type.lower()
        
        logger.info(f"Loading {model_type.upper()} model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        try:
            if self.model_type == "clip":
                # Load CLIP model and processor
                self.model = CLIPModel.from_pretrained(
                    model_name, 
                    cache_dir=cache_dir
                ).to(self.device)
                
                self.processor = CLIPProcessor.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
                
                # Get embedding dimension
                self.embedding_dim = self.model.config.projection_dim
                
            elif self.model_type == "siglip":
                # Load SigLIP model and processor
                self.model = SiglipModel.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                ).to(self.device)
                
                self.processor = SiglipProcessor.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
                
                # Get embedding dimension
                self.embedding_dim = self.model.config.vision_config.hidden_size
                
            else:
                raise ValueError(f"Unknown model type: {model_type}. Use 'clip' or 'siglip'")
            
            self.model.eval()  # Set to evaluation mode
            logger.info("Model loaded successfully")
            logger.info(f"Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    @torch.no_grad()
    def encode_image(self, 
                     images: Union[Image.Image, List[Image.Image], str, List[str]],
                     normalize: bool = True) -> np.ndarray:
        """
        Encode images to embeddings
        
        Args:
            images: Single image, list of images, or image path(s)
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            Image embeddings array of shape (N, embedding_dim)
        """
        # Convert to list if single image
        if isinstance(images, (str, Image.Image)):
            images = [images]
        
        # Load images if paths
        loaded_images = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            loaded_images.append(img)
        
        try:
            # Process images
            inputs = self.processor(
                images=loaded_images,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Get image features
            image_features = self.model.get_image_features(**inputs)
            
            # Normalize if requested
            if normalize:
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Convert to numpy
            embeddings = image_features.cpu().numpy()
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding images: {e}")
            raise
    
    @torch.no_grad()
    def encode_text(self,
                   texts: Union[str, List[str]],
                   normalize: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings
        
        Args:
            texts: Single text or list of texts
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            Text embeddings array of shape (N, embedding_dim)
        """
        # Convert to list if single text
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Process texts
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77  # CLIP max length
            ).to(self.device)
            
            # Get text features
            text_features = self.model.get_text_features(**inputs)
            
            # Normalize if requested
            if normalize:
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Convert to numpy
            embeddings = text_features.cpu().numpy()
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise
    
    @torch.no_grad()
    def compute_similarity(self,
                          image_embeddings: np.ndarray,
                          text_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute similarity between image and text embeddings
        
        Args:
            image_embeddings: Array of shape (N, embedding_dim)
            text_embeddings: Array of shape (M, embedding_dim)
            
        Returns:
            Similarity matrix of shape (N, M)
        """
        # Compute cosine similarity
        similarity = np.dot(image_embeddings, text_embeddings.T)
        
        return similarity
    
    def encode_multimodal(self,
                         image: Union[Image.Image, str],
                         text: str,
                         fusion_method: str = "concat") -> np.ndarray:
        """
        Encode image and text together
        
        Args:
            image: Image or image path
            text: Text description
            fusion_method: How to combine embeddings
                - "concat": Concatenate embeddings
                - "average": Average embeddings
                - "weighted": Weighted average (0.5 each)
            
        Returns:
            Combined embedding
        """
        # Get individual embeddings
        img_emb = self.encode_image(image, normalize=True)[0]
        txt_emb = self.encode_text(text, normalize=True)[0]
        
        if fusion_method == "concat":
            combined = np.concatenate([img_emb, txt_emb])
        elif fusion_method == "average":
            combined = (img_emb + txt_emb) / 2
        elif fusion_method == "weighted":
            # Can adjust weights based on task
            combined = 0.5 * img_emb + 0.5 * txt_emb
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Re-normalize
        combined = combined / np.linalg.norm(combined)
        
        return combined
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim


class EnsembleEncoder:
    """
    Ensemble multiple encoders for robust features
    Useful for combining different pre-trained models
    """
    
    def __init__(self, encoders: List[MedCLIPEncoder], weights: Optional[List[float]] = None):
        """
        Args:
            encoders: List of MedCLIP encoders
            weights: Weights for each encoder (default: equal weights)
        """
        self.encoders = encoders
        
        if weights is None:
            self.weights = [1.0 / len(encoders)] * len(encoders)
        else:
            assert len(weights) == len(encoders)
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    @torch.no_grad()
    def encode_image(self, images: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        """Encode images using ensemble"""
        embeddings_list = []
        
        for encoder, weight in zip(self.encoders, self.weights):
            emb = encoder.encode_image(images, normalize=True)
            embeddings_list.append(emb * weight)
        
        # Weighted sum
        ensemble_emb = np.sum(embeddings_list, axis=0)
        
        # Re-normalize
        ensemble_emb = ensemble_emb / np.linalg.norm(ensemble_emb, axis=-1, keepdims=True)
        
        return ensemble_emb
    
    @torch.no_grad()
    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts using ensemble"""
        embeddings_list = []
        
        for encoder, weight in zip(self.encoders, self.weights):
            emb = encoder.encode_text(texts, normalize=True)
            embeddings_list.append(emb * weight)
        
        # Weighted sum
        ensemble_emb = np.sum(embeddings_list, axis=0)
        
        # Re-normalize
        ensemble_emb = ensemble_emb / np.linalg.norm(ensemble_emb, axis=-1, keepdims=True)
        
        return ensemble_emb


def test_encoder():
    """Test the MedCLIP encoder"""
    print("=== Testing MedCLIP Encoder ===\n")
    
    # Initialize encoder
    encoder = MedCLIPEncoder(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Test text encoding
    print("Testing text encoding...")
    texts = [
        "Chest X-ray showing pneumonia in right lower lobe",
        "No acute cardiopulmonary abnormality",
        "Pleural effusion in left hemithorax"
    ]
    text_embeddings = encoder.encode_text(texts)
    print(f"Text embeddings shape: {text_embeddings.shape}")
    
    # Test image encoding
    print("\nTesting image encoding...")
    # Create dummy image
    dummy_images = [Image.new('RGB', (224, 224), color='white') for _ in range(2)]
    image_embeddings = encoder.encode_image(dummy_images)
    print(f"Image embeddings shape: {image_embeddings.shape}")
    
    # Test similarity
    print("\nTesting similarity computation...")
    similarity = encoder.compute_similarity(image_embeddings[:1], text_embeddings)
    print(f"Similarity matrix shape: {similarity.shape}")
    print(f"Similarity scores: {similarity}")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_encoder()
