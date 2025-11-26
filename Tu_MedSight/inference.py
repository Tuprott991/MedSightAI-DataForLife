import os
import sys
import argparse
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from io import BytesIO
import cv2
import pydicom

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import MedicalConceptModel


class MedicalConceptInference:
    """Inference class for Medical Concept Model with interpretability."""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"[Inference] Using device: {self.device}")
        
        # Load checkpoint
        print(f"[Inference] Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.config = checkpoint['config']
        self.concept_names = checkpoint['concept_names']
        self.disease_names = checkpoint['disease_names']
        
        # Build model
        self.model = MedicalConceptModel(
            model_name=self.config['backbone_name'],
            num_concepts=len(self.concept_names),
            num_classes=self.config['num_classes'],
            projection_dim=self.config['projection_dim'],
            prototypes_per_concept=self.config['prototypes_per_concept'],
            freeze_backbone=True,
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.config['image_size'], self.config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        print(f"[Inference] Model loaded successfully")
        print(f"[Inference] Concepts: {len(self.concept_names)}")
        print(f"[Inference] Diseases: {len(self.disease_names)}")
    
    def load_image(self, image_path):
        """
        Load image from various formats (PNG, JPG, DICOM).
        
        Args:
            image_path: Path to image file or PIL Image
            
        Returns:
            PIL Image in RGB
        """
        if isinstance(image_path, Image.Image):
            image = image_path.convert('RGB')
        elif isinstance(image_path, str):
            # Try loading as DICOM first (most common for medical images)
            try:
                dcm = pydicom.dcmread(image_path)
                pixel_array = dcm.pixel_array
                
                # Handle Photometric Interpretation (invert if MONOCHROME1)
                if hasattr(dcm, 'PhotometricInterpretation'):
                    if dcm.PhotometricInterpretation == 'MONOCHROME1':
                        pixel_array = np.amax(pixel_array) - pixel_array
                
                # Normalize to 0-255
                pixel_array = pixel_array - np.min(pixel_array)
                pixel_array = pixel_array / (np.max(pixel_array) + 1e-8) * 255
                pixel_array = pixel_array.astype(np.uint8)
                
                # Convert to RGB
                if len(pixel_array.shape) == 2:
                    pixel_array = np.stack([pixel_array] * 3, axis=-1)
                
                image = Image.fromarray(pixel_array)
            except Exception as e:
                # Fallback to regular image loading (PNG, JPG)
                try:
                    image = Image.open(image_path).convert('RGB')
                except Exception as e2:
                    print(f"[Error] Failed to load image: {e}, {e2}")
                    # Return black image
                    image = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
        else:
            # Assume numpy array
            if len(image_path.shape) == 2:
                image_path = np.stack([image_path] * 3, axis=-1)
            image = Image.fromarray(image_path.astype(np.uint8))
        
        return image
    
    @torch.no_grad()
    def predict(self, image, disabled_concepts=None):
        """
        Run inference on an image.
        
        Args:
            image: PIL Image or path to image
            disabled_concepts: List of concept indices to disable (set to 0)
            
        Returns:
            Dictionary with predictions and interpretability outputs
        """
        # Load and preprocess image
        pil_image = self.load_image(image)
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Forward pass
        outputs = self.model(image_tensor, return_all=True)
        
        # Get predictions
        class_logits = outputs['class_logits']  # (1, num_classes)
        class_probs = torch.sigmoid(class_logits).cpu().numpy()[0]
        
        # Concept predictions (similarity scores)
        sim_k_agg = outputs['sim_k_agg'].cpu().numpy()[0]  # (K,)
        
        # Apply concept masking if specified
        if disabled_concepts is not None and len(disabled_concepts) > 0:
            sim_k_masked = sim_k_agg.copy()
            sim_k_masked[disabled_concepts] = 0.0
            
            # Re-predict with masked concepts
            sim_k_tensor = torch.from_numpy(sim_k_masked).unsqueeze(0).to(self.device)
            class_logits_masked = self.model.classifier(sim_k_tensor)
            class_probs_masked = torch.sigmoid(class_logits_masked).cpu().numpy()[0]
        else:
            class_probs_masked = class_probs
        
        # Get CAMs and prototypes
        cams = outputs['cams'].cpu().numpy()[0]  # (K, H, W)
        cam_weights = outputs['cam_weights'].cpu().numpy()[0]  # (K, H, W)
        sim_maps = outputs['sim_maps'].cpu().numpy()[0]  # (K, M, H, W)
        prototypes = outputs['prototypes'].cpu().numpy()  # (K, M, P)
        
        result = {
            'pil_image': pil_image,
            'class_probs': class_probs,
            'class_probs_masked': class_probs_masked,
            'concept_sims': sim_k_agg,
            'cams': cams,
            'cam_weights': cam_weights,
            'sim_maps': sim_maps,
            'prototypes': prototypes,
        }
        
        return result
    
    def visualize_cam(self, image, cam, alpha=0.5, colormap='jet'):
        """
        Overlay CAM on original image.
        
        Args:
            image: PIL Image
            cam: 2D numpy array (H, W)
            alpha: Overlay transparency
            colormap: Matplotlib colormap name
            
        Returns:
            PIL Image with CAM overlay
        """
        # Resize image to match CAM size
        image_np = np.array(image.resize((cam.shape[1], cam.shape[0])))
        
        # Normalize CAM to [0, 1]
        cam_normalized = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Apply colormap
        cmap = cm.get_cmap(colormap)
        cam_colored = cmap(cam_normalized)[:, :, :3]  # RGB
        cam_colored = (cam_colored * 255).astype(np.uint8)
        
        # Blend with original image
        blended = cv2.addWeighted(image_np, 1 - alpha, cam_colored, alpha, 0)
        
        return Image.fromarray(blended)
    
    def visualize_prototype(self, prototype_vector, sim_map):
        """
        Visualize prototype as a heatmap and similarity map.
        
        Args:
            prototype_vector: (P,) prototype embedding
            sim_map: (H, W) similarity map for this prototype
            
        Returns:
            PIL Image showing similarity map
        """
        # Normalize similarity map
        sim_normalized = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-8)
        
        # Apply colormap
        cmap = cm.get_cmap('viridis')
        sim_colored = cmap(sim_normalized)[:, :, :3]
        sim_colored = (sim_colored * 255).astype(np.uint8)
        
        # Resize for better visualization
        sim_colored = cv2.resize(sim_colored, (224, 224), interpolation=cv2.INTER_NEAREST)
        
        return Image.fromarray(sim_colored)
    
    def get_top_prototype(self, concept_idx, sim_maps):
        """
        Get the top prototype for a concept based on max similarity.
        
        Args:
            concept_idx: Concept index
            sim_maps: (K, M, H, W) similarity maps
            
        Returns:
            prototype_idx: Index of top prototype
            max_sim: Maximum similarity value
        """
        concept_sims = sim_maps[concept_idx]  # (M, H, W)
        max_sims = concept_sims.reshape(concept_sims.shape[0], -1).max(axis=1)  # (M,)
        prototype_idx = max_sims.argmax()
        max_sim = max_sims[prototype_idx]
        
        return prototype_idx, max_sim


def create_gradio_interface(checkpoint_path):
    """Create Gradio interface for inference."""
    
    # Initialize inference
    inferencer = MedicalConceptInference(checkpoint_path)
    
    def predict_and_visualize(image, *concept_checkboxes):
        """
        Main prediction function for Gradio.
        
        Args:
            image: Input image from Gradio
            *concept_checkboxes: Boolean flags for each concept (enabled/disabled)
            
        Returns:
            Tuple of outputs for Gradio interface
        """
        if image is None:
            return None, "Please upload an image", [], None
        
        # Get disabled concepts
        disabled_concepts = [
            i for i, enabled in enumerate(concept_checkboxes) if not enabled
        ]
        
        # Run inference
        result = inferencer.predict(image, disabled_concepts=disabled_concepts)
        
        # Prepare outputs
        outputs = []
        
        # 1. Display original image
        original_image = result['pil_image']
        
        # 2. Concept predictions with CAMs and prototypes
        concept_outputs = []
        for concept_idx in range(len(inferencer.concept_names)):
            concept_name = inferencer.concept_names[concept_idx]
            concept_sim = result['concept_sims'][concept_idx]
            
            # Check if concept is disabled
            is_disabled = concept_idx in disabled_concepts
            status = "❌ DISABLED" if is_disabled else f"✓ Similarity: {concept_sim:.3f}"
            
            # CAM visualization
            cam = result['cam_weights'][concept_idx]
            cam_overlay = inferencer.visualize_cam(original_image, cam, alpha=0.5)
            
            # Top prototype visualization
            prototype_idx, max_sim = inferencer.get_top_prototype(
                concept_idx, result['sim_maps']
            )
            sim_map = result['sim_maps'][concept_idx, prototype_idx]
            prototype_vis = inferencer.visualize_prototype(
                result['prototypes'][concept_idx, prototype_idx],
                sim_map
            )
            
            concept_outputs.append({
                'name': concept_name,
                'status': status,
                'cam': cam_overlay,
                'prototype': prototype_vis,
                'similarity': concept_sim,
            })
        
        # 3. Disease predictions
        disease_probs_original = result['class_probs']
        disease_probs_masked = result['class_probs_masked']
        
        # Format disease predictions as text
        disease_text = "### Disease Predictions\n\n"
        
        if inferencer.config['num_classes'] == 1:
            # Binary classification
            disease_text += f"**{inferencer.disease_names[0]}**\n\n"
            disease_text += f"- Original probability: **{disease_probs_original[0]:.1%}**\n"
            if len(disabled_concepts) > 0:
                disease_text += f"- With concept masking: **{disease_probs_masked[0]:.1%}**\n"
                change = disease_probs_masked[0] - disease_probs_original[0]
                disease_text += f"- Change: **{change:+.1%}**\n"
        else:
            # Multi-class
            for i, disease_name in enumerate(inferencer.disease_names[:inferencer.config['num_classes']]):
                prob_orig = disease_probs_original[i]
                prob_masked = disease_probs_masked[i]
                
                disease_text += f"**{disease_name}**\n"
                disease_text += f"- Original: {prob_orig:.1%}\n"
                if len(disabled_concepts) > 0:
                    disease_text += f"- Masked: {prob_masked:.1%} ({prob_masked - prob_orig:+.1%})\n"
                disease_text += "\n"
        
        # 4. Create concept gallery
        concept_gallery = []
        concept_info_text = "### Concept Analysis\n\n"
        
        for i, output in enumerate(concept_outputs):
            concept_info_text += f"**{i+1}. {output['name']}**\n"
            concept_info_text += f"- {output['status']}\n\n"
            
            # Add to gallery (CAM and prototype side by side)
            concept_gallery.append((output['cam'], f"{output['name']} - CAM"))
            concept_gallery.append((output['prototype'], f"{output['name']} - Prototype"))
        
        return original_image, disease_text, concept_gallery, concept_info_text
    
    # Build Gradio interface
    with gr.Blocks(title="Medical Concept Model Inference", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🏥 Medical Concept Model - Interpretable Chest X-ray Analysis")
        gr.Markdown(
            "Upload a chest X-ray image (PNG, JPG, or DICOM) to analyze medical concepts "
            "and disease predictions. You can disable specific concepts to see their impact on diagnosis."
        )
        
        with gr.Row():
            # Left column: Input
            with gr.Column(scale=1):
                gr.Markdown("## 📥 Input")
                
                image_input = gr.Image(
                    type="pil",
                    label="Upload Chest X-ray",
                    height=400
                )
                
                gr.Markdown("### Concept Selection")
                gr.Markdown("Uncheck concepts to disable them in diagnosis:")
                
                # Create checkboxes for each concept
                concept_checkboxes = []
                for concept_name in inferencer.concept_names:
                    checkbox = gr.Checkbox(
                        label=concept_name,
                        value=True,
                        interactive=True,
                    )
                    concept_checkboxes.append(checkbox)
                
                predict_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
            
            # Right column: Outputs
            with gr.Column(scale=2):
                gr.Markdown("## 📊 Results")
                
                with gr.Tab("Disease Diagnosis"):
                    disease_output = gr.Markdown()
                
                with gr.Tab("Concept Visualization"):
                    concept_info = gr.Markdown()
                    concept_gallery = gr.Gallery(
                        label="Concept CAMs and Prototypes",
                        columns=4,
                        height="auto",
                        object_fit="contain"
                    )
                
                with gr.Tab("Original Image"):
                    original_output = gr.Image(label="Original Image")
        
        # Examples
        gr.Markdown("## 📋 Examples")
        gr.Markdown("Try these example images:")
        
        example_images = [
            # Add paths to example images here
            # ["path/to/example1.png"],
            # ["path/to/example2.dcm"],
        ]
        
        # Connect predict button
        predict_btn.click(
            fn=predict_and_visualize,
            inputs=[image_input] + concept_checkboxes,
            outputs=[original_output, disease_output, concept_gallery, concept_info]
        )
        
        # Auto-predict on image upload
        image_input.change(
            fn=predict_and_visualize,
            inputs=[image_input] + concept_checkboxes,
            outputs=[original_output, disease_output, concept_gallery, concept_info]
        )
    
    return demo


def main():
    parser = argparse.ArgumentParser(description='Medical Concept Model Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--share', action='store_true',
                        help='Create public share link')
    parser.add_argument('--port', type=int, default=7860,
                        help='Port for Gradio interface')
    
    args = parser.parse_args()
    
    # Create and launch interface
    print("=" * 80)
    print("Creating Gradio Interface")
    print("=" * 80)
    
    demo = create_gradio_interface(args.checkpoint)
    
    print("\n" + "=" * 80)
    print("Launching Gradio Interface")
    print("=" * 80)
    
    demo.launch(
        share=args.share,
        server_port=args.port,
        inbrowser=True,
    )


if __name__ == '__main__':
    main()
