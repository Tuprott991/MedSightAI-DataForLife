from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from disease_detection.inference import predict_image, load_image_from_bytes  # noqa: E402


CLASS_ID_TO_VI: dict[int, str] = {
    0: "Gian rong dong mach chu",
    1: "Xep phoi",
    2: "Voi hoa",
    3: "Tim to",
    4: "Dong dac phoi",
    5: "Benh phoi ke",
    6: "Tham nhiem",
    7: "Dam mo phoi",
    8: "Not/Khoi",
    9: "Ton thuong khac",
    10: "Tran dich mang phoi",
    11: "Day mang phoi",
    12: "Tran khi mang phoi",
    13: "Xo phoi",
}


class DiseaseDetectionService:
    def analyze_image(
        self,
        image_bytes: bytes,
        *,
        stage: int = 2,
        img_size: int = 640,
        wbf_iou: float = 0.4,
        score_thres: float = 0.25,
        device: str = "",
    ) -> tuple[list[dict[str, Any]], bytes]:
        image_bgr = load_image_from_bytes(image_bytes)
        result = predict_image(
            image_bgr=image_bgr,
            stage=stage,
            img_size=img_size,
            wbf_iou=wbf_iou,
            score_thres=score_thres,
            device=device,
        )

        detections = [self._normalize_detection(detection) for detection in result["detections"]]
        rendered_image = result["rendered_image_bgr"]
        success, encoded = cv2.imencode(".jpg", rendered_image)
        if not success:
            raise RuntimeError("Failed to encode disease detection output image")

        return detections, encoded.tobytes()

    @staticmethod
    def _normalize_detection(detection: dict[str, Any]) -> dict[str, Any]:
        class_id = int(detection["class_id"])
        class_name_en = str(detection["class_name"])
        return {
            "class_id": class_id,
            "class_name_en": class_name_en,
            "class_name_vi": CLASS_ID_TO_VI.get(class_id, class_name_en),
            "confidence": float(detection["confidence"]),
            "x1": int(detection["x1"]),
            "y1": int(detection["y1"]),
            "x2": int(detection["x2"]),
            "y2": int(detection["y2"]),
        }


disease_detection_service = DiseaseDetectionService()
