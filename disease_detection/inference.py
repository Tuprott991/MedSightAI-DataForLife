import base64
import io
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import onnxruntime as ort
from ensemble_boxes import weighted_boxes_fusion


CUR_PATH = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = CUR_PATH / "weights"
DEFAULT_FOLDS = [0, 1, 2, 3, 4]
DEFAULT_TTA = [0, 4]

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

BOX_COLOR = (0, 0, 255)
TEXT_COLOR = (0, 0, 255)
TEXT_BG_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.9
FONT_THICKNESS = 2
BOX_THICKNESS = 3


def image_rot(image, factor):
    return np.rot90(image, factor)


def flip_hor(image):
    return np.fliplr(image)


def bbox_rot90(bbox, factor, height, width):
    if factor not in {0, 1, 2, 3}:
        raise ValueError("factor must be in {0, 1, 2, 3}")
    x_min, y_min, x_max, y_max = bbox[:4]
    if factor == 1:
        bbox = y_min, width - x_max, y_max, width - x_min
    elif factor == 2:
        bbox = width - x_max, height - y_max, width - x_min, height - y_min
    elif factor == 3:
        bbox = height - y_max, x_min, height - y_min, x_max
    return bbox


def flip_hor_boxes(bbox, width):
    x_min, y_min, x_max, y_max = bbox[:4]
    return width - x_max, y_min, width - x_min, y_max


def get_tta_pair(ind):
    if ind == 0:
        return lambda image: image, lambda box, height, width: box
    if ind == 1:
        return lambda image: image_rot(image, 1), lambda box, height, width: bbox_rot90(box, 3, height, width)
    if ind == 2:
        return lambda image: image_rot(image, 2), lambda box, height, width: bbox_rot90(box, 2, height, width)
    if ind == 3:
        return lambda image: image_rot(image, 3), lambda box, height, width: bbox_rot90(box, 1, height, width)
    if ind == 4:
        return flip_hor, lambda box, height, width: flip_hor_boxes(box, width)
    if ind == 5:
        return (
            lambda image: image_rot(flip_hor(image), 1),
            lambda box, height, width: flip_hor_boxes(bbox_rot90(box, 3, height, width), height),
        )
    if ind == 6:
        return (
            lambda image: image_rot(flip_hor(image), 2),
            lambda box, height, width: flip_hor_boxes(bbox_rot90(box, 2, height, width), width),
        )
    if ind == 7:
        return (
            lambda image: image_rot(flip_hor(image), 3),
            lambda box, height, width: flip_hor_boxes(bbox_rot90(box, 1, height, width), height),
        )
    raise ValueError(f"Unsupported TTA id: {ind}")


def letterbox(image_bgr, new_shape=640, color=(114, 114, 114)):
    shape = image_bgr.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image_bgr = cv2.resize(image_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image_bgr = cv2.copyMakeBorder(
        image_bgr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return image_bgr, ratio, (dw, dh)


def xywh2xyxy(boxes):
    converted = boxes.copy()
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def clip_boxes(boxes, image_shape):
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, image_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, image_shape[0])
    return boxes


def scale_coords(boxes, image_shape, ratio, pad):
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= ratio
    return clip_boxes(boxes, image_shape)


def box_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-9
    return inter / union


def nms_numpy(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou_thres]

    return np.array(keep, dtype=np.int64)


def non_max_suppression_numpy(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    agnostic=False,
    max_det=300,
    max_wh=4096,
):
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


@lru_cache(maxsize=16)
def create_session(weights_path: str, device: str):
    available = ort.get_available_providers()
    if device.lower() == "cpu":
        providers = ["CPUExecutionProvider"]
    elif device.lower() in {"cuda", "gpu"} and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif device and device.lower() not in {"", "cpu", "cuda", "gpu"}:
        raise ValueError(f"Unsupported device: {device}. Use '', 'cpu', or 'cuda'.")
    else:
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )

    return ort.InferenceSession(weights_path, providers=providers)


def prepare_input(image_bgr, img_size):
    processed, ratio, pad = letterbox(image_bgr, new_shape=img_size)
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    processed = processed.transpose(2, 0, 1).astype(np.float32) / 255.0
    processed = np.expand_dims(processed, axis=0)
    return processed, ratio, pad


def run_session(session, image_bgr, img_size, conf_thres=0.01, iou_thres=0.4):
    input_tensor, ratio, pad = prepare_input(image_bgr, img_size)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor})[0]

    detections = non_max_suppression_numpy(
        output, conf_thres=conf_thres, iou_thres=iou_thres
    )[0]
    if len(detections):
        detections[:, :4] = scale_coords(detections[:, :4], image_bgr.shape[:2], ratio, pad).round()

    if len(detections) == 0:
        return (
            np.empty((0, 4), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    boxes = detections[:, :4].astype(np.int32)
    scores = detections[:, 4].astype(np.float32)
    labels = detections[:, 5].astype(np.int32)
    return boxes, scores, labels


def run_single_tta(session, image_bgr, image_size, aug_ind):
    image_tta, box_reverse_tta = get_tta_pair(aug_ind)
    transformed_image = np.ascontiguousarray(image_tta(image_bgr))
    h, w = transformed_image.shape[:2]

    boxes, scores, labels = run_session(
        session, transformed_image, image_size, conf_thres=0.01, iou_thres=0.4
    )
    if len(boxes) > 0:
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h)
        for idx in range(boxes.shape[0]):
            boxes[idx, :] = box_reverse_tta(boxes[idx, :], h, w)

    return boxes, scores, labels


def merge_predictions(predictions, iou_thr):
    boxes_list = []
    scores_list = []
    labels_list = []
    max_value = 10000.0

    for boxes, scores, labels in predictions:
        boxes = np.array(boxes, copy=False)
        scores = np.array(scores, copy=False)
        labels = np.array(labels, copy=False)
        boxes_list.append(boxes / max_value)
        scores_list.append(scores)
        labels_list.append(labels)

    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=None,
        iou_thr=iou_thr,
        skip_box_thr=0.0,
    )
    boxes = np.round(boxes * max_value).astype(np.int32)
    return boxes, scores, labels.astype(np.int32)


def draw_detections(image_bgr, boxes, scores, labels):
    rendered = image_bgr.copy()
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(rendered, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        class_name = CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else str(label)
        text = f"{label} {class_name}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, FONT, FONT_SCALE, FONT_THICKNESS
        )
        text_y = max(text_height + 6, y1 - 10)
        bg_tl = (x1, max(0, text_y - text_height - baseline - 6))
        bg_br = (x1 + text_width + 8, text_y + baseline - 2)
        cv2.rectangle(rendered, bg_tl, bg_br, TEXT_BG_COLOR, -1)
        cv2.putText(
            rendered,
            text,
            (x1 + 4, text_y - 4),
            FONT,
            FONT_SCALE,
            TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )
    return rendered


def encode_image_b64(image_bgr, suffix=".jpg"):
    ext = ".png" if suffix.lower().endswith(".png") else ".jpg"
    success, encoded = cv2.imencode(ext, image_bgr)
    if not success:
        raise RuntimeError("Failed to encode rendered image")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def load_image_from_bytes(image_bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode uploaded image")
    return image


def get_onnx_path(stage, fold):
    path = WEIGHTS_DIR / f"stage{stage}_fold{fold}.onnx"
    if not path.exists():
        raise FileNotFoundError(f"Missing ONNX weights: {path}")
    return path


def predict_image(
    image_bgr,
    stage=2,
    folds: Iterable[int] = DEFAULT_FOLDS,
    tta: Iterable[int] = DEFAULT_TTA,
    img_size=640,
    wbf_iou=0.4,
    score_thres=0.1,
    device="",
):
    if image_bgr is None:
        raise ValueError("Input image is empty or unreadable")

    folds = list(folds)
    tta = list(tta)

    all_predictions = []
    sessions_used = []
    for fold in folds:
        session = create_session(str(get_onnx_path(stage, fold)), device)
        sessions_used.append({"fold": fold, "providers": session.get_providers()})
        for aug_ind in tta:
            all_predictions.append(run_single_tta(session, image_bgr, img_size, aug_ind))

    boxes, scores, labels = merge_predictions(all_predictions, wbf_iou)
    keep = scores >= score_thres
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    detections = []
    for box, score, label in zip(boxes, scores, labels):
        detections.append(
            {
                "class_id": int(label),
                "class_name": CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else str(label),
                "confidence": float(score),
                "x1": int(box[0]),
                "y1": int(box[1]),
                "x2": int(box[2]),
                "y2": int(box[3]),
            }
        )

    rendered = draw_detections(image_bgr, boxes, scores, labels)
    return {
        "stage": stage,
        "folds": folds,
        "tta": tta,
        "img_size": img_size,
        "wbf_iou": wbf_iou,
        "score_thres": score_thres,
        "detections": detections,
        "providers": sessions_used,
        "rendered_image_bgr": rendered,
    }
