"""
Visualize Concept Activation Maps (CAMs) to see where concepts are located.

Usage:
    python visualize_cams.py --checkpoint best_model_stage1.pth --image path/to/xray.png
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import argparse
from pathlib import Path

from src.model import CSR

def visualize_concept_heatmaps(model, image_path, concept_names=None, save_dir='outputs'):
    """
    Visualize CAMs for all concepts in an X-ray image.
    
    Args:
        model: Trained CSR model
        image_path: Path to X-ray image
        concept_names: List of concept names (e.g., ['Lung Opacity', 'Cardiomegaly'])
        save_dir: Directory to save visualization
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_pil = Image.open(image_path).convert('RGB')
    img_tensor = transform(img_pil).unsqueeze(0).to(device)  # (1, 3, 224, 224)
    
    # Get heatmaps
    heatmaps = model.get_concept_heatmaps(img_tensor, apply_sigmoid=True)  # (1, K, 224, 224)
    heatmaps = heatmaps.squeeze(0).cpu().numpy()  # (K, 224, 224)
    
    # Original image for overlay
    img_np = np.array(img_pil.resize((224, 224)))
    
    # Create visualization
    K = heatmaps.shape[0]
    cols = 4
    rows = (K + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten() if K > 1 else [axes]
    
    for k in range(K):
        ax = axes[k]
        
        # Show original image
        ax.imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)
        
        # Overlay heatmap
        heatmap = heatmaps[k]
        im = ax.imshow(heatmap, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        
        # Title
        title = concept_names[k] if concept_names and k < len(concept_names) else f'Concept {k}'
        max_activation = heatmap.max()
        ax.set_title(f'{title}\n(max: {max_activation:.3f})', fontsize=10)
        ax.axis('off')
        
        # Colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for k in range(K, len(axes)):
        axes[k].axis('off')
    
    plt.tight_layout()
    
    # Save
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    output_path = save_dir / f'{Path(image_path).stem}_cams.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'✅ Saved visualization to: {output_path}')
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize Concept Activation Maps')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to X-ray image')
    parser.add_argument('--num_concepts', type=int, default=22, help='Number of concepts')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of disease classes')
    parser.add_argument('--backbone', type=str, default='medmae', help='Backbone type')
    parser.add_argument('--weights', type=str, default='weights/pre_trained_medmae.pth', help='MedMAE weights')
    parser.add_argument('--save_dir', type=str, default='outputs', help='Output directory')
    
    args = parser.parse_args()
    
    # VinDr-CXR concept names (22 anatomical findings)
    concept_names = [
        'Aortic enlargement', 'Atelectasis', 'Calcification',
        'Cardiomegaly', 'Clavicle fracture', 'Consolidation',
        'Edema', 'Emphysema', 'Enlarged PA',
        'ILD', 'Infiltration', 'Lung Opacity',
        'Lung cavity', 'Lung cyst', 'Mediastinal shift',
        'Nodule/Mass', 'Pleural effusion', 'Pleural thickening',
        'Pneumothorax', 'Pulmonary fibrosis', 'Rib fracture',
        'Other lesion'
    ]
    
    # Load model
    print(f'Loading model from {args.checkpoint}...')
    model = CSR(
        num_concepts=args.num_concepts,
        num_classes=args.num_classes,
        num_prototypes_per_concept=1,
        backbone_type=args.backbone,
        model_name=args.weights
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint)
    
    # Visualize
    print(f'Visualizing CAMs for {args.image}...')
    visualize_concept_heatmaps(model, args.image, concept_names[:args.num_concepts], args.save_dir)


if __name__ == '__main__':
    main()
