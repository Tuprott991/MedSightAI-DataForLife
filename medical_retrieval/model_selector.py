"""
Utility to get model configuration
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from config import MODEL_CONFIG


def get_active_model_config():
    """Get configuration for the active model"""
    active_model = MODEL_CONFIG.get('active_model', 'biomedclip')
    
    if active_model not in MODEL_CONFIG:
        raise ValueError(f"Active model '{active_model}' not found in MODEL_CONFIG")
    
    config = MODEL_CONFIG[active_model].copy()
    config['active_model'] = active_model
    
    return config


def print_model_info():
    """Print information about available models"""
    print("=" * 80)
    print("Available Models:")
    print("=" * 80)
    
    models_info = {
        'pubmed_clip': {
            'name': 'PubMed CLIP',
            'pros': ['Fast', 'Good for general medical'],
            'cons': ['Lower accuracy than BiomedCLIP'],
            'use_case': 'Quick prototyping, real-time inference'
        },
        'biomedclip': {
            'name': 'BiomedCLIP',
            'pros': ['Good medical accuracy', 'Pre-trained on PubMed'],
            'cons': ['Slightly slower than PubMed CLIP', 'Not specialized for CXR'],
            'use_case': 'General medical imaging, multi-modal tasks'
        },
        'siglip': {
            'name': 'SigLIP (Base)',
            'pros': ['Better loss function', 'Scalable', 'State-of-the-art architecture'],
            'cons': ['Needs fine-tuning on medical data', '768-dim (larger)'],
            'use_case': 'Research, custom fine-tuning'
        },
        'medsiglip': {
            'name': 'MedSigLIP-448 (VinDR-CXR)',
            'pros': ['BEST for Chest X-rays', 'Fine-tuned on VinDR dataset', 'High resolution (448x448)', 'Lowest loss', 'SOTA performance'],
            'cons': ['Larger model (1152-dim)', 'Slower than smaller models', 'Requires more memory'],
            'use_case': 'PRODUCTION - Chest X-ray retrieval (RECOMMENDED)'
        }
    }
    
    active = MODEL_CONFIG.get('active_model', 'biomedclip')
    
    for key, info in models_info.items():
        status = "ACTIVE" if key == active else "  "
        print(f"\n{status} {info['name']} ({key}):")
        print(f"  Pros: {', '.join(info['pros'])}")
        print(f"  Cons: {', '.join(info['cons'])}")
        print(f"  Use Case: {info['use_case']}")
        
        if key in MODEL_CONFIG:
            print(f"  Model: {MODEL_CONFIG[key]['model_name']}")
            print(f"  Embedding Dim: {MODEL_CONFIG[key]['embedding_dim']}")
    
    print("\n" + "=" * 80)
    print(f"Current Active Model: {active}")
    print("=" * 80)
    print("\nRECOMMENDATION:")
    print("  For Chest X-ray retrieval: Use 'medsiglip' (BEST performance)")
    print("  For general medical imaging: Use 'biomedclip'")
    print("  For fast prototyping: Use 'pubmed_clip'")
    print("\nTo change active model, edit config.py:")
    print("  MODEL_CONFIG['active_model'] = 'medsiglip'  # Recommended for CXR")
    print()


if __name__ == "__main__":
    print_model_info()
    
    print("\nActive Model Config:")
    print("-" * 80)
    config = get_active_model_config()
    for key, value in config.items():
        print(f"  {key}: {value}")
