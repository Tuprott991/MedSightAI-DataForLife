"""
CSR Model Inference Script
Implements similarity reasoning as described in the paper:

"For an unseen image x, CSR calculates the similarity scores between x and 
the examples (or prototypes) of each concept. It then uses the concept 
similarity scores to support the prediction of y."

Inference Flow (Figure 2):
1. Feature Extraction: x → F → f  (backbone)
2. Projection: f → P → f'  (projector)
3. Similarity Computation: cos_sim(f'(h,w), p_k^m) → S  (conv with prototypes)
4. Max Pooling: S → s = [s_k^m]  (max similarity per prototype)
5. Task Prediction: s → H → y  (task head)

For interpretability:
- CAMs: C(f) shows which concepts are activated
- Similarity maps: S shows where each prototype matches in the image
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import pydicom
import cv2
import argparse
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model import CSRModel


def load_image(image_path, target_size=(224, 224)):
    """
    Load DICOM or PNG/JPG image and preprocess.
    
    Args:
        image_path: Path to image
        target_size: Target size (H, W)
    
    Returns:
        image_tensor: (1, 3, H, W) normalized tensor
        original_image: PIL Image for visualization
    """
    if image_path.lower().endswith('.dcm'):
        # Load DICOM
        dcm = pydicom.dcmread(image_path)
        pixel_array = dcm.pixel_array.astype(np.float32)
        
        # Normalize to 0-255
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
        pixel_array = pixel_array.astype(np.uint8)
        
        # Convert to RGB (replicate grayscale to 3 channels)
        if len(pixel_array.shape) == 2:
            pixel_array = np.stack([pixel_array] * 3, axis=-1)
        
        image = Image.fromarray(pixel_array)
    else:
        # Load regular image
        image = Image.open(image_path).convert('RGB')
    
    # Resize
    original_image = image.copy()
    image = image.resize(target_size)
    
    # To tensor and normalize (ImageNet stats)
    image_array = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = (image_array - mean) / std
    
    # (H, W, C) -> (C, H, W)
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    
    return image_tensor, original_image


def inference(model, image_tensor, device, mode='full'):
    """
    CSR Inference following the paper's logic:
    
    1. Extract features f from backbone F
    2. Project features f' = P(f) 
    3. Compute similarity maps S between each patch f'(h,w) and prototypes p
    4. Max pooling on S to get similarity scores s = [s_k^m] for k=1..K, m=1..M
    5. Predict disease y = H(s) using task head
    
    For visualization:
    - CAMs from concept head C(f)
    - Similarity maps S for each prototype
    
    Args:
        model: CSR model
        image_tensor: (1, 3, H, W) input tensor
        device: torch device
        mode: 'full', 'cams_only', or 'prototypes_only'
    
    Returns:
        results: Dict with predictions and heatmaps
    """
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        if mode == 'cams_only':
            # Stage 1 only: Concept Activation Maps (for training Stage 1)
            # Only use backbone + concept_head
            f = model.backbone(image_tensor)  # (1, 768, 14, 14)
            cams = model.concept_head(f)  # (1, K, 14, 14)
            
            # Get concept probabilities (max pool + sigmoid)
            concept_logits = F.adaptive_max_pool2d(cams, (1, 1)).squeeze(-1).squeeze(-1)  # (1, K)
            concept_probs = torch.sigmoid(concept_logits).squeeze(0).cpu().numpy()  # (K,)
            
            # Upsample CAMs to input size for visualization
            cams_upsampled = F.interpolate(cams, size=(224, 224), mode='bilinear', align_corners=False)
            heatmaps = torch.sigmoid(cams_upsampled).squeeze(0).cpu().numpy()  # (K, 224, 224)
            
            return {
                'concept_probs': concept_probs,
                'concept_heatmaps': heatmaps,
                'similarity_scores': None,
                'similarity_maps': None,
                'disease_probs': None
            }
        
        elif mode == 'prototypes_only':
            # Stages 1+2: Concepts + Prototypes (for training Stage 2)
            # Use full forward but don't interpret task head output
            outputs = model(image_tensor)
            
            # Concept visualization
            cams = outputs['cams']  # (1, K, 14, 14)
            concept_logits = F.adaptive_max_pool2d(cams, (1, 1)).squeeze(-1).squeeze(-1)
            concept_probs = torch.sigmoid(concept_logits).squeeze(0).cpu().numpy()
            
            cams_upsampled = F.interpolate(cams, size=(224, 224), mode='bilinear', align_corners=False)
            heatmaps = torch.sigmoid(cams_upsampled).squeeze(0).cpu().numpy()
            
            # Similarity scores: (1, K*M) -> reshape to (K, M) for better interpretation
            similarity_scores = outputs['similarity_scores'].squeeze(0).cpu().numpy()  # (K*M,)
            K = model.K
            M = model.M
            similarity_scores_per_concept = similarity_scores.reshape(K, M)  # (K, M)
            
            # Max similarity per concept (across M prototypes)
            concept_similarity = similarity_scores_per_concept.max(axis=1)  # (K,)
            
            # Upsample similarity maps for visualization
            similarity_maps = outputs['similarity_maps']  # (1, K*M, 14, 14)
            similarity_maps_upsampled = F.interpolate(similarity_maps, size=(224, 224), 
                                                     mode='bilinear', align_corners=False)
            similarity_maps_np = similarity_maps_upsampled.squeeze(0).cpu().numpy()  # (K*M, 224, 224)
            
            return {
                'concept_probs': concept_probs,
                'concept_heatmaps': heatmaps,
                'similarity_scores': similarity_scores_per_concept,  # (K, M)
                'concept_similarity': concept_similarity,  # (K,) max across prototypes
                'similarity_maps': similarity_maps_np,  # (K*M, 224, 224)
                'disease_probs': None
            }
        
        else:  # full
            # Full CSR inference: All 3 stages (Fig. 2 in paper)
            # x -> F -> f -> P -> f' -> similarity(f', p) -> S -> max -> s -> H -> y
            outputs = model(image_tensor)
            
            # 1. Concept Activation Maps (for interpretability)
            cams = outputs['cams']  # (1, K, 14, 14)
            concept_logits = F.adaptive_max_pool2d(cams, (1, 1)).squeeze(-1).squeeze(-1)
            concept_probs = torch.sigmoid(concept_logits).squeeze(0).cpu().numpy()
            
            cams_upsampled = F.interpolate(cams, size=(224, 224), mode='bilinear', align_corners=False)
            heatmaps = torch.sigmoid(cams_upsampled).squeeze(0).cpu().numpy()
            
            # 2. Similarity Scores (Eq. 1 in paper)
            # s = [s_k^m] where s_k^m = max_{h,w} cos_sim(f'(h,w), p_k^m)
            similarity_scores = outputs['similarity_scores'].squeeze(0).cpu().numpy()  # (K*M,)
            K = model.K
            M = model.M
            similarity_scores_per_concept = similarity_scores.reshape(K, M)  # (K, M)
            concept_similarity = similarity_scores_per_concept.max(axis=1)  # (K,)
            
            # 3. Similarity Maps (for visualization)
            similarity_maps = outputs['similarity_maps']  # (1, K*M, 14, 14)
            similarity_maps_upsampled = F.interpolate(similarity_maps, size=(224, 224), 
                                                     mode='bilinear', align_corners=False)
            similarity_maps_np = similarity_maps_upsampled.squeeze(0).cpu().numpy()
            
            # 4. Disease Predictions: y = H(s)
            logits = outputs['logits']  # (1, num_classes)
            disease_probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            
            return {
                'concept_probs': concept_probs,  # From CAMs (for visualization)
                'concept_heatmaps': heatmaps,  # CAM heatmaps
                'similarity_scores': similarity_scores_per_concept,  # (K, M) similarity to each prototype
                'concept_similarity': concept_similarity,  # (K,) max similarity per concept
                'similarity_maps': similarity_maps_np,  # (K*M, 224, 224) spatial similarity maps
                'disease_probs': disease_probs  # Final disease predictions
            }


def visualize_results(original_image, results, concept_names, disease_names, 
                     mode='full', threshold=0.5, top_k=5, save_path=None):
    """
    Visualize inference results.
    
    Args:
        original_image: PIL Image
        results: Dict from inference()
        concept_names: List of K concept names
        disease_names: List of disease names
        mode: Visualization mode
        threshold: Confidence threshold for display
        top_k: Number of top concepts/diseases to show
        save_path: Path to save visualization
    """
    # Resize original image to 224x224 for overlay
    original_image_resized = original_image.resize((224, 224))
    original_array = np.array(original_image_resized).astype(np.float32) / 255.0
    
    if mode == 'cams_only':
        # Show top-K concept heatmaps
        concept_probs = results['concept_probs']
        heatmaps = results['concept_heatmaps']
        
        # Get top-K concepts above threshold
        top_indices = np.argsort(concept_probs)[::-1][:top_k]
        top_indices = [i for i in top_indices if concept_probs[i] > threshold]
        
        if len(top_indices) == 0:
            print(f"No concepts above threshold {threshold}")
            return
        
        # Create figure
        n_cols = min(len(top_indices), 3)
        n_rows = (len(top_indices) + n_cols - 1) // n_cols + 1  # +1 for original
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Hide extra subplots in first row
        for j in range(1, n_cols):
            axes[0, j].axis('off')
        
        # Concept heatmaps
        for plot_idx, concept_idx in enumerate(top_indices):
            row = 1 + plot_idx // n_cols
            col = plot_idx % n_cols
            
            # Overlay heatmap on original image
            heatmap = heatmaps[concept_idx]
            heatmap_colored = cm.jet(heatmap)[:, :, :3]  # RGB
            
            overlay = 0.6 * original_array + 0.4 * heatmap_colored
            overlay = np.clip(overlay, 0, 1)
            
            axes[row, col].imshow(overlay)
            axes[row, col].set_title(
                f'{concept_names[concept_idx]}\nConfidence: {concept_probs[concept_idx]:.3f}',
                fontsize=12
            )
            axes[row, col].axis('off')
        
        # Hide extra subplots
        for plot_idx in range(len(top_indices), n_rows * n_cols - n_cols):
            row = 1 + plot_idx // n_cols
            col = plot_idx % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
    elif mode == 'prototypes_only':
        # Show concepts + similarity reasoning
        concept_probs = results['concept_probs']
        heatmaps = results['concept_heatmaps']
        similarity_scores = results['similarity_scores']  # (K, M)
        concept_similarity = results['concept_similarity']  # (K,)
        similarity_maps = results['similarity_maps']  # (K*M, 224, 224)
        
        K, M = similarity_scores.shape
        
        # Get top-K concepts by similarity
        top_indices = np.argsort(concept_similarity)[::-1][:top_k]
        top_indices = [i for i in top_indices if concept_similarity[i] > threshold]
        
        if len(top_indices) == 0:
            print(f"No concepts above threshold {threshold}")
            return
        
        # Create figure: Original | Concept CAM | Top Prototype Similarity Map
        n_concepts = len(top_indices)
        fig, axes = plt.subplots(n_concepts + 1, 3, figsize=(15, 5*(n_concepts+1)))
        if n_concepts == 0:
            axes = axes.reshape(1, -1)
        
        # Row 0: Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        axes[0, 1].axis('off')
        axes[0, 2].axis('off')
        
        # Rows 1+: Each concept
        for plot_idx, concept_idx in enumerate(top_indices):
            row = plot_idx + 1
            
            # Column 0: Concept name + CAM confidence
            axes[row, 0].imshow(original_array)
            axes[row, 0].set_title(
                f'Concept: {concept_names[concept_idx]}\n'
                f'CAM Confidence: {concept_probs[concept_idx]:.3f}\n'
                f'Max Similarity: {concept_similarity[concept_idx]:.3f}',
                fontsize=12, fontweight='bold'
            )
            axes[row, 0].axis('off')
            
            # Column 1: Concept CAM heatmap
            heatmap = heatmaps[concept_idx]
            heatmap_colored = cm.jet(heatmap)[:, :, :3]
            overlay = 0.6 * original_array + 0.4 * heatmap_colored
            overlay = np.clip(overlay, 0, 1)
            axes[row, 1].imshow(overlay)
            axes[row, 1].set_title('Concept Activation Map', fontsize=11)
            axes[row, 1].axis('off')
            
            # Column 2: Best prototype similarity map
            prototype_sims = similarity_scores[concept_idx]  # (M,)
            best_prototype_idx = np.argmax(prototype_sims)
            global_prototype_idx = concept_idx * M + best_prototype_idx
            
            sim_map = similarity_maps[global_prototype_idx]
            sim_map_colored = cm.viridis(sim_map)[:, :, :3]
            overlay_sim = 0.6 * original_array + 0.4 * sim_map_colored
            overlay_sim = np.clip(overlay_sim, 0, 1)
            axes[row, 2].imshow(overlay_sim)
            axes[row, 2].set_title(
                f'Prototype {best_prototype_idx+1}/{M}\n'
                f'Similarity: {prototype_sims[best_prototype_idx]:.3f}',
                fontsize=11
            )
            axes[row, 2].axis('off')
        
        plt.tight_layout()
        
    else:  # full
        # Show complete reasoning chain: Concepts → Similarity → Disease Predictions
        concept_probs = results['concept_probs']
        heatmaps = results['concept_heatmaps']
        similarity_scores = results['similarity_scores']  # (K, M)
        concept_similarity = results['concept_similarity']  # (K,)
        similarity_maps = results['similarity_maps']  # (K*M, 224, 224)
        disease_probs = results['disease_probs']
        
        K, M = similarity_scores.shape
        
        # Get top concepts and diseases
        top_concept_indices = np.argsort(concept_similarity)[::-1][:top_k]
        top_concept_indices = [i for i in top_concept_indices if concept_similarity[i] > threshold]
        
        top_disease_indices = np.argsort(disease_probs)[::-1][:top_k]
        top_disease_indices = [i for i in top_disease_indices if disease_probs[i] > threshold]
        
        if len(top_concept_indices) == 0:
            print(f"No concepts above threshold {threshold}")
            return
        
        # Create figure
        n_concepts = len(top_concept_indices)
        n_rows = n_concepts + 2  # +1 for original, +1 for disease predictions
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        
        # Row 0: Original image + Disease predictions
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Disease predictions bar chart
        if len(top_disease_indices) > 0:
            disease_names_top = [disease_names[i] for i in top_disease_indices]
            disease_probs_top = [disease_probs[i] for i in top_disease_indices]
            
            axes[0, 1].barh(range(len(disease_names_top)), disease_probs_top, color='steelblue')
            axes[0, 1].set_yticks(range(len(disease_names_top)))
            axes[0, 1].set_yticklabels(disease_names_top, fontsize=10)
            axes[0, 1].set_xlabel('Probability', fontsize=12)
            axes[0, 1].set_title('Disease Predictions (Task Head)', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlim(0, 1)
            axes[0, 1].invert_yaxis()
            
            # Add probability values
            for i, (idx, prob) in enumerate(zip(top_disease_indices, disease_probs_top)):
                axes[0, 1].text(prob + 0.02, i, f'{prob:.3f}', va='center', fontsize=10)
        else:
            axes[0, 1].text(0.5, 0.5, f'No diseases above threshold {threshold}', 
                           ha='center', va='center', fontsize=12)
            axes[0, 1].axis('off')
        
        axes[0, 2].axis('off')
        
        # Row 1: Header
        axes[1, 0].text(0.5, 0.5, 'Concept Info', ha='center', va='center', 
                       fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        axes[1, 1].text(0.5, 0.5, 'Concept Activation (CAM)', ha='center', va='center',
                       fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        axes[1, 2].text(0.5, 0.5, 'Prototype Similarity', ha='center', va='center',
                       fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
        
        # Rows 2+: Each concept
        for plot_idx, concept_idx in enumerate(top_concept_indices):
            row = plot_idx + 2
            
            # Column 0: Concept info
            info_text = f'Concept: {concept_names[concept_idx]}\n\n'
            info_text += f'CAM Confidence: {concept_probs[concept_idx]:.3f}\n'
            info_text += f'Max Similarity: {concept_similarity[concept_idx]:.3f}\n\n'
            info_text += 'Prototype Similarities:\n'
            for m in range(M):
                info_text += f'  P{m+1}: {similarity_scores[concept_idx, m]:.3f}\n'
            
            axes[row, 0].text(0.1, 0.5, info_text, ha='left', va='center', 
                            fontsize=11, family='monospace')
            axes[row, 0].axis('off')
            
            # Column 1: Concept CAM
            heatmap = heatmaps[concept_idx]
            heatmap_colored = cm.jet(heatmap)[:, :, :3]
            overlay = 0.6 * original_array + 0.4 * heatmap_colored
            overlay = np.clip(overlay, 0, 1)
            axes[row, 1].imshow(overlay)
            axes[row, 1].axis('off')
            
            # Column 2: Best prototype similarity map
            prototype_sims = similarity_scores[concept_idx]
            best_prototype_idx = np.argmax(prototype_sims)
            global_prototype_idx = concept_idx * M + best_prototype_idx
            
            sim_map = similarity_maps[global_prototype_idx]
            sim_map_colored = cm.viridis(sim_map)[:, :, :3]
            overlay_sim = 0.6 * original_array + 0.4 * sim_map_colored
            overlay_sim = np.clip(overlay_sim, 0, 1)
            axes[row, 2].imshow(overlay_sim)
            axes[row, 2].set_title(f'Best Prototype: {best_prototype_idx+1}/{M}', fontsize=11)
            axes[row, 2].axis('off')
        
        plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='CSR Model Inference with Similarity Reasoning')
    parser.add_argument('--image', type=str, required=True, help='Path to input image (DICOM or PNG/JPG)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['full', 'cams_only', 'prototypes_only'],
                       help='Inference mode (default: full)')
    parser.add_argument('--K', type=int, default=10, help='Number of concepts')
    parser.add_argument('--M', type=int, default=5, help='Number of prototypes per concept')
    parser.add_argument('--num_classes', type=int, default=14, help='Number of disease classes')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for display')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to show')
    parser.add_argument('--output', type=str, default=None, help='Path to save visualization')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = CSRModel(K=args.K, M=args.M, num_classes=args.num_classes)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Load image
    print(f"Loading image from {args.image}")
    image_tensor, original_image = load_image(args.image)
    print(f"Image shape: {image_tensor.shape}")
    
    # Run inference
    print(f"Running inference in mode: {args.mode}")
    results = inference(model, image_tensor, device, mode=args.mode)
    
    # Print results
    print("\n" + "="*60)
    print("INFERENCE RESULTS")
    print("="*60)
    
    if results['concept_probs'] is not None:
        print("\nConcept Predictions:")
        for i, prob in enumerate(results['concept_probs']):
            if prob > args.threshold:
                print(f"  Concept {i}: {prob:.4f}")
    
    if results.get('concept_similarity') is not None:
        print("\nConcept Similarity Scores (Max across prototypes):")
        for i, sim in enumerate(results['concept_similarity']):
            if sim > args.threshold:
                print(f"  Concept {i}: {sim:.4f}")
    
    if results['disease_probs'] is not None:
        print("\nDisease Predictions:")
        for i, prob in enumerate(results['disease_probs']):
            if prob > args.threshold:
                print(f"  Disease {i}: {prob:.4f}")
    
    print("="*60)
    
    # Create concept and disease names
    concept_names = [f'Concept {i}' for i in range(args.K)]
    disease_names = [f'Disease {i}' for i in range(args.num_classes)]
    
    # Visualize
    print("\nGenerating visualization...")
    visualize_results(
        original_image, 
        results, 
        concept_names, 
        disease_names,
        mode=args.mode,
        threshold=args.threshold,
        top_k=args.top_k,
        save_path=args.output
    )
    
    print("Inference complete!")


if __name__ == '__main__':
    main()
