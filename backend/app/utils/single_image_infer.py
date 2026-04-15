"""
Chest X-ray lesion localization using YOLOv5 ONNX model.

Model: Long5504/yolov5-chest-lesion-localize (model.onnx)
Source: https://huggingface.co/Long5504/yolov5-chest-lesion-localize

Public API
----------
run_localization(image_bytes: bytes) -> tuple[list[dict], bytes]
    Run inference on raw image bytes.
    Returns (detections, annotated_jpeg_bytes).

    Each detection dict:
        {
            "class_id":      int,
            "class_name_en": str,   # English — drawn on image
            "class_name_vi": str,   # Vietnamese — shown in UI
            "confidence":    float,
            "x1": int, "y1": int,
            "x2": int, "y2": int,
        }
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import threading
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CUR_PATH = Path(__file__).resolve().parent

HF_REPO_ID = "Long5504/yolov5-chest-lesion-localize"
HF_FILENAME = "model.onnx"

# Default local cache directory (can be overridden via MEDSIGHT_MODEL_CACHE env var)
_DEFAULT_CACHE_DIR = Path(
    os.environ.get("MEDSIGHT_MODEL_CACHE", Path.home() / ".cache" / "medsight" / "models")
)

CLASS_NAMES: List[str] = [
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

CLASS_ID_TO_VI: dict[int, str] = {
    0: "Giãn rộng động mạch chủ",
    1: "Xẹp phổi",
    2: "Vôi hóa",
    3: "Tim to",
    4: "Đông đặc phổi",
    5: "Bệnh phổi kẽ",
    6: "Thâm nhiễm",
    7: "Đám mờ phổi",
    8: "Nốt/Khối",
    9: "Tổn thương khác",
    10: "Tràn dịch màng phổi",
    11: "Dày màng phổi",
    12: "Tràn khí màng phổi",
    13: "Xơ phổi",
}

# Bounding-box colour palette per class (BGR for OpenCV)
_COLOURS: List[Tuple[int, int, int]] = [
    (0, 200, 255), (0, 255, 128), (255, 100, 0), (255, 50, 200),
    (0, 128, 255), (200, 255, 0), (255, 180, 0), (0, 255, 200),
    (255, 0, 100), (128, 0, 255), (0, 220, 220), (220, 0, 220),
    (255, 128, 50), (50, 200, 255),
]


def _colour_for(class_id: int) -> Tuple[int, int, int]:
    return _COLOURS[class_id % len(_COLOURS)]


# ---------------------------------------------------------------------------
# Lazy ONNX session singleton
# ---------------------------------------------------------------------------

_session: ort.InferenceSession | None = None
_session_lock = threading.Lock()


def _download_model(cache_dir: Path) -> Path:
    """Download model.onnx from HuggingFace if not already cached."""
    from huggingface_hub import hf_hub_download  # imported lazily

    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / HF_FILENAME
    if model_path.exists():
        logger.info("[Localize] Using cached ONNX model at %s", model_path)
        return model_path

    logger.info("[Localize] Downloading ONNX model from HuggingFace: %s / %s", HF_REPO_ID, HF_FILENAME)
    downloaded = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        local_dir=str(cache_dir),
    )
    logger.info("[Localize] Model downloaded to %s", downloaded)
    return Path(downloaded)


def get_yolo_session(
    device: str = "",
    cache_dir: Path | None = None,
) -> ort.InferenceSession:
    """
    Return the global ONNX InferenceSession, downloading the model if needed.
    Thread-safe via double-checked locking.

    Args:
        device: "", "cpu", or "cuda"
        cache_dir: Override for model cache directory.
    """
    global _session
    if _session is not None:
        return _session

    with _session_lock:
        if _session is not None:  # re-check after acquiring lock
            return _session

        resolved_cache = cache_dir or _DEFAULT_CACHE_DIR
        model_path = _download_model(resolved_cache)
        _session = create_session(model_path, device)
        logger.info(
            "[Localize] ONNX session ready. Providers: %s",
            _session.get_providers(),
        )
    return _session


# ---------------------------------------------------------------------------
# Image pre/post-processing helpers
# ---------------------------------------------------------------------------


def letterbox(
    image_bgr: np.ndarray,
    new_shape: int | Tuple[int, int] = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resize + pad to square while preserving aspect ratio."""
    shape = image_bgr.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        image_bgr = cv2.resize(image_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image_bgr = cv2.copyMakeBorder(
        image_bgr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return image_bgr, ratio, (dw, dh)


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = boxes.copy()
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def clip_boxes(boxes: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, image_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, image_shape[0])
    return boxes


def scale_coords(
    boxes: np.ndarray,
    image_shape: Tuple[int, int],
    ratio: float,
    pad: Tuple[float, float],
) -> np.ndarray:
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= ratio
    return clip_boxes(boxes, image_shape)


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-9
    return inter / union


def nms_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thres: float,
) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)

    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou_thres]
    return np.array(keep, dtype=np.int64)


def non_max_suppression_numpy(
    prediction: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    agnostic: bool = False,
    max_det: int = 300,
    max_wh: int = 4096,
) -> List[np.ndarray]:
    outputs = []
    for pred in prediction:
        pred = pred[pred[:, 4] > conf_thres]
        if pred.shape[0] == 0:
            outputs.append(np.zeros((0, 6), dtype=np.float32))
            continue

        pred[:, 5:] *= pred[:, 4:5]
        boxes = xywh2xyxy(pred[:, :4])
        class_scores = pred[:, 5:]
        class_ids = class_scores.argmax(axis=1)
        confidences = class_scores[np.arange(class_scores.shape[0]), class_ids]

        keep_mask = confidences > conf_thres
        boxes = boxes[keep_mask]
        confidences = confidences[keep_mask]
        class_ids = class_ids[keep_mask]

        if boxes.shape[0] == 0:
            outputs.append(np.zeros((0, 6), dtype=np.float32))
            continue

        offsets = 0 if agnostic else class_ids.astype(np.float32) * max_wh
        nms_boxes = boxes.copy()
        nms_boxes[:, [0, 2]] += offsets[:, None]
        keep = nms_numpy(nms_boxes, confidences, iou_thres)[:max_det]

        detections = np.concatenate(
            [
                boxes[keep],
                confidences[keep, None].astype(np.float32),
                class_ids[keep, None].astype(np.float32),
            ],
            axis=1,
        )
        outputs.append(detections)
    return outputs


def create_session(weights_path: Path, device: str = "") -> ort.InferenceSession:
    """Create an ONNX InferenceSession, preferring CUDA when available."""
    available = ort.get_available_providers()
    if device.lower() == "cpu":
        providers = ["CPUExecutionProvider"]
    elif device.lower() in {"cuda", "gpu"} and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif device and device not in {"", "cpu", "cuda", "gpu"}:
        raise ValueError(f"Unsupported device: {device!r}. Use '', 'cpu', or 'cuda'.")
    else:
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
    return ort.InferenceSession(str(weights_path), providers=providers)


def prepare_input(
    image_bgr: np.ndarray,
    img_size: int,
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    processed, ratio, pad = letterbox(image_bgr, new_shape=img_size)
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    processed = processed.transpose(2, 0, 1).astype(np.float32) / 255.0
    processed = np.expand_dims(processed, axis=0)
    return processed, ratio, pad


def run_inference(
    session: ort.InferenceSession,
    image_bgr: np.ndarray,
    img_size: int = 640,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
) -> List[dict]:
    """
    Run YOLO inference and return a list of detection dicts.

    Each dict:
        class_id, class_name_en, class_name_vi, confidence, x1, y1, x2, y2
    """
    input_tensor, ratio, pad = prepare_input(image_bgr, img_size)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor})[0]

    detections = non_max_suppression_numpy(
        output, conf_thres=conf_thres, iou_thres=iou_thres
    )[0]

    if len(detections):
        detections[:, :4] = scale_coords(
            detections[:, :4], image_bgr.shape[:2], ratio, pad
        ).round()

    results: List[dict] = []
    for x1, y1, x2, y2, conf, cls in detections.tolist():
        cls = int(cls)
        results.append(
            {
                "class_id": cls,
                "class_name_en": CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls),
                "class_name_vi": CLASS_ID_TO_VI.get(cls, str(cls)),
                "confidence": round(float(conf), 4),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
            }
        )
    return results


def draw_detections(
    image_bgr: np.ndarray,
    detections: List[dict],
    thickness: int = 2,
    font_scale: float = 0.55,
) -> np.ndarray:
    """
    Draw bounding boxes + English class labels on a copy of the image.
    Returns the annotated BGR array.
    """
    rendered = image_bgr.copy()
    h, w = rendered.shape[:2]

    for det in detections:
        colour = _colour_for(det["class_id"])
        pt1 = (det["x1"], det["y1"])
        pt2 = (det["x2"], det["y2"])
        cv2.rectangle(rendered, pt1, pt2, colour, thickness)

        label = f'{det["class_name_en"]} {det["confidence"]:.2f}'
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        # Background pill above bbox
        lx = det["x1"]
        ly = max(det["y1"] - th - baseline - 4, 0)
        cv2.rectangle(rendered, (lx, ly), (lx + tw + 4, ly + th + baseline + 4), colour, cv2.FILLED)
        cv2.putText(
            rendered,
            label,
            (lx + 2, ly + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return rendered


# ---------------------------------------------------------------------------
# Public high-level API
# ---------------------------------------------------------------------------


def run_localization(
    image_bytes: bytes,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    img_size: int = 640,
) -> Tuple[List[dict], bytes]:
    """
    Run YOLOv5 chest-lesion localization on raw image bytes.

    Downloads the model on first call (thread-safe). Subsequent calls reuse
    the cached ONNX session.

    Args:
        image_bytes: Raw image bytes (JPEG, PNG, etc.)
        conf_thres:  Confidence scoring threshold (default 0.25)
        iou_thres:   NMS IoU threshold (default 0.45)
        img_size:    Model input resolution (default 640)

    Returns:
        detections:           List of detection dicts (class_id, class_name_en,
                              class_name_vi, confidence, x1, y1, x2, y2)
        annotated_jpeg_bytes: JPEG bytes of the image with bboxes drawn (English labels)
    """
    # Decode image
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Failed to decode image bytes — unsupported format or corrupt data.")

    session = get_yolo_session()
    detections = run_inference(session, image_bgr, img_size, conf_thres, iou_thres)
    annotated_bgr = draw_detections(image_bgr, detections)

    # Encode annotated image to JPEG bytes
    ok, jpeg_buf = cv2.imencode(
        ".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92]
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed for annotated image.")

    annotated_bytes = jpeg_buf.tobytes()
    logger.info("[Localize] Detected %d lesion(s).", len(detections))
    return detections, annotated_bytes


# ---------------------------------------------------------------------------
# CLI entry-point (unchanged from original)
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOv5 chest-lesion localization.")
    parser.add_argument("--weights", type=Path, help="Path to an .onnx checkpoint (skips HuggingFace download)")
    parser.add_argument("--image", required=True, type=Path, help="Path to an input image")
    parser.add_argument("--img-size", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", default="", help="Use '', 'cpu', or 'cuda'")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CUR_PATH / "runs" / "single-image",
        help="Output directory",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(args.image))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    # Use provided weights or auto-download
    if args.weights:
        if args.weights.suffix.lower() != ".onnx":
            raise ValueError(f"--weights must be an .onnx file, got: {args.weights}")
        session = create_session(args.weights, args.device)
    else:
        session = get_yolo_session(device=args.device)

    detections = run_inference(
        session=session,
        image_bgr=image_bgr,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
    )

    stem = args.image.stem
    rendered = draw_detections(image_bgr, detections)
    out_image = args.output_dir / f"{stem}_pred.jpg"
    out_json = args.output_dir / f"{stem}_pred.json"

    cv2.imwrite(str(out_image), rendered)
    out_json.write_text(json.dumps(detections, indent=2, ensure_ascii=False))

    print(f"providers={session.get_providers()}")
    print(f"saved_image={out_image}")
    print(f"saved_json={out_json}")
    print(f"detections={len(detections)}")
    for det in detections:
        print(
            f'{det["class_id"]}\t{det["class_name_en"]}\t{det["class_name_vi"]}'
            f'\t{det["confidence"]:.4f}\t{det["x1"]},{det["y1"]},{det["x2"]},{det["y2"]}'
        )


if __name__ == "__main__":
    main()
