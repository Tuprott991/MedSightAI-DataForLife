import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from service import load_csr_model, preprocess_image, infer_cams


CLASS_NAMES = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
]


def _normalize_cam(cam: np.ndarray, percentile: float) -> np.ndarray:
    cam = np.asarray(cam, dtype=np.float32)
    cam = np.maximum(cam, 0.0)
    upper = float(np.percentile(cam, percentile))
    if upper <= 1e-8:
        upper = float(cam.max())
    if upper <= 1e-8:
        return np.zeros_like(cam, dtype=np.float32)
    cam = np.clip(cam / upper, 0.0, 1.0)
    return cam


def overlay_cam(
    image_path: str,
    cam: np.ndarray,
    output_path: Path,
    alpha: float,
    blur_ksize: int,
    percentile: float,
    gamma: float,
) -> None:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    cam_resized = cv2.resize(np.asarray(cam, dtype=np.float32), (image.shape[1], image.shape[0]))
    if blur_ksize > 1:
        cam_resized = cv2.GaussianBlur(cam_resized, (blur_ksize, blur_ksize), 0)
    cam_norm = _normalize_cam(cam_resized, percentile=percentile)
    if gamma != 1.0:
        cam_norm = np.power(cam_norm, gamma)

    heatmap = cv2.applyColorMap((cam_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blend = (cam_norm * alpha)[..., None]
    overlay = image_rgb * (1.0 - blend) + heatmap * blend

    output_bgr = cv2.cvtColor((overlay * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), output_bgr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run csr_phase1.pth on a single chest X-ray and save CAM overlays.")
    parser.add_argument("--image", required=True, help="Path to input PNG/JPG image")
    parser.add_argument("--checkpoint", default=str(Path(__file__).resolve().parent.parent / "csr_phase1.pth"))
    parser.add_argument("--output-dir", default="outputs/csr_phase1")
    parser.add_argument("--threshold", type=float, default=0.5, help="Minimum prediction probability to keep a class")
    parser.add_argument("--alpha", type=float, default=0.65, help="Maximum heatmap opacity")
    parser.add_argument("--blur-ksize", type=int, default=41, help="Gaussian blur kernel size for smoother saliency")
    parser.add_argument("--percentile", type=float, default=99.5, help="Percentile used to normalize CAM intensity")
    parser.add_argument("--gamma", type=float, default=0.7, help="Gamma for emphasizing mid/high saliency regions")
    args = parser.parse_args()

    if args.blur_ksize % 2 == 0:
        raise ValueError("--blur-ksize must be an odd number")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_csr_model(args.checkpoint, device)
    image_tensor = preprocess_image(args.image).to(device)

    probs, cams = infer_cams(model, image_tensor, device)

    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    top_concept = CLASS_NAMES[top_idx]

    results = []
    if top_prob >= args.threshold:
        overlay_path = output_dir / f"{top_idx:02d}_{top_concept.replace('/', '_').replace(' ', '_')}.png"
        overlay_cam(
            image_path=args.image,
            cam=cams[top_idx],
            output_path=overlay_path,
            alpha=args.alpha,
            blur_ksize=args.blur_ksize,
            percentile=args.percentile,
            gamma=args.gamma,
        )
        results.append(
            {
                "class_idx": top_idx,
                "concept": top_concept,
                "probability": top_prob,
                "overlay_path": str(overlay_path.resolve()),
            }
        )

    summary = {
        "image": str(Path(args.image).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": str(device),
        "threshold": args.threshold,
        "alpha": args.alpha,
        "blur_ksize": args.blur_ksize,
        "percentile": args.percentile,
        "gamma": args.gamma,
        "top_class_idx": top_idx,
        "top_concept": top_concept,
        "top_probability": top_prob,
        "detections": results,
    }

    summary_path = output_dir / "results.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nSaved overlays and JSON summary to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
