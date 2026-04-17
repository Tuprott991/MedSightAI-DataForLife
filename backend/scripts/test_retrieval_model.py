"""
Smoke-test the local ONNX retrieval model on one image.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def load_retrieval_service():
    module_path = BACKEND_DIR / "app" / "services" / "ai_service.py"
    spec = importlib.util.spec_from_file_location("retrieval_ai_service", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load retrieval service module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.retrieval_embedding_service


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the ONNX retrieval model.")
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "test.png",
        help="Path to an input image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    retrieval_embedding_service = load_retrieval_service()
    image_bytes = args.image.read_bytes()
    embedding = retrieval_embedding_service.generate_image_embedding(image_bytes)
    model_info = retrieval_embedding_service.get_model_info()

    print("Retrieval model smoke test passed")
    print(f"image={args.image}")
    print(f"embedding_dim={len(embedding)}")
    print(f"input_size={model_info['input_size']}")
    print(f"providers={model_info['providers']}")
    print(f"first_values={embedding[:5]}")


if __name__ == "__main__":
    main()
