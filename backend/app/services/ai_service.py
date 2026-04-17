"""
AI and retrieval model services.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort
from fastapi import HTTPException
from PIL import Image

from app.config.settings import settings


# Add model paths to system path for other team modules.
sys.path.append(os.path.abspath(settings.MODEL_INFERENCE_PATH))
sys.path.append(os.path.abspath(settings.MEDGEMMA_PATH))
sys.path.append(os.path.abspath(settings.VINDR_DATASET_PATH))


try:
    _BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    _BICUBIC = Image.BICUBIC


class AIModelService:
    """Placeholder service for the main diagnostic model stack."""

    def __init__(self):
        pass

    def preprocess_image(self, image_path: str) -> str:
        raise NotImplementedError("Connect to MedSightAI preprocessing module")

    def run_inference(self, image_path: str) -> Dict[str, Any]:
        raise NotImplementedError("Connect to MedSightAI inference module")

    def generate_gradcam(self, image_path: str, target_layer: str = "features.denseblock4") -> str:
        raise NotImplementedError("Implement Grad-CAM generation")

    def extract_concepts(self, image_path: str, bounding_boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raise NotImplementedError("Implement concept extraction")


class ImageRetrievalService:
    """ONNX-backed image embedding service for CBIR retrieval."""

    def __init__(self):
        self.session: ort.InferenceSession | None = None
        self.input_name: str | None = None
        self.output_name: str | None = None
        self.input_size: int | None = None
        self.embedding_dim: int | None = None
        self.providers: List[str] = []
        self._initialized = False

    @staticmethod
    def _resolve_model_path() -> Path:
        return Path(settings.RETRIEVAL_MODEL_PATH).expanduser().resolve()

    @staticmethod
    def _select_providers() -> List[str]:
        available = ort.get_available_providers()
        providers: List[str] = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        if "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")
        return providers or available

    @staticmethod
    def _resolve_spatial_size(shape: List[Any]) -> int:
        spatial_dims = [value for value in shape[-2:] if isinstance(value, int) and value > 0]
        if not spatial_dims:
            raise HTTPException(status_code=500, detail=f"Unsupported ONNX input shape: {shape}")
        if spatial_dims[0] != spatial_dims[-1]:
            raise HTTPException(status_code=500, detail=f"Expected square ONNX input shape, got: {shape}")
        return spatial_dims[0]

    @staticmethod
    def _resolve_embedding_dim(shape: List[Any]) -> int:
        for value in reversed(shape):
            if isinstance(value, int) and value > 0:
                return value
        raise HTTPException(status_code=500, detail=f"Unsupported ONNX output shape: {shape}")

    def _lazy_load(self) -> None:
        if self._initialized:
            return

        model_path = self._resolve_model_path()
        if not model_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Retrieval model not found at {model_path}",
            )

        try:
            self.providers = self._select_providers()
            self.session = ort.InferenceSession(str(model_path), providers=self.providers)

            input_meta = self.session.get_inputs()[0]
            output_meta = self.session.get_outputs()[0]

            self.input_name = input_meta.name
            self.output_name = output_meta.name
            self.input_size = self._resolve_spatial_size(input_meta.shape)
            self.embedding_dim = self._resolve_embedding_dim(output_meta.shape)

            if self.embedding_dim != settings.ZILLIZ_IMG_DIMENSION:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Retrieval model embedding dimension does not match "
                        f"ZILLIZ_IMG_DIMENSION ({self.embedding_dim} != {settings.ZILLIZ_IMG_DIMENSION})"
                    ),
                )

            self._initialized = True
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load retrieval model: {exc}",
            ) from exc

    def _prepare_image(self, image_bytes: bytes) -> np.ndarray:
        if self.input_size is None:
            raise HTTPException(status_code=500, detail="Retrieval model is not initialized")

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image = self._resize_shortest_side(image, settings.RETRIEVAL_RESIZE_SIZE)
            image = self._center_crop(image, settings.RETRIEVAL_IMAGE_SIZE)

            if settings.RETRIEVAL_IMAGE_SIZE != self.input_size:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Configured retrieval crop size does not match the ONNX model input size "
                        f"({settings.RETRIEVAL_IMAGE_SIZE} != {self.input_size}). "
                        "Update RETRIEVAL_IMAGE_SIZE or re-export the ONNX model with 384x384 input."
                    ),
                )

            image_array = np.asarray(image, dtype=np.float32) / 255.0
            image_array = (image_array - 0.5) / 0.5
            image_array = np.transpose(image_array, (2, 0, 1))
            return np.expand_dims(image_array, axis=0).astype(np.float32, copy=False)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to preprocess retrieval image: {exc}",
            ) from exc

    @staticmethod
    def _resize_shortest_side(image: Image.Image, resize_size: int) -> Image.Image:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size: {(width, height)}")

        if width < height:
            new_width = resize_size
            new_height = round(height * (resize_size / width))
        else:
            new_height = resize_size
            new_width = round(width * (resize_size / height))

        return image.resize((new_width, new_height), _BICUBIC)

    @staticmethod
    def _center_crop(image: Image.Image, crop_size: int) -> Image.Image:
        width, height = image.size
        if crop_size > width or crop_size > height:
            raise ValueError(
                f"Center crop size {crop_size} is larger than resized image {(width, height)}"
            )

        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        return image.crop((left, top, right, bottom))

    def get_model_info(self) -> Dict[str, Any]:
        self._lazy_load()
        model_path = self._resolve_model_path()
        return {
            "model_path": str(model_path),
            "providers": self.providers,
            "input_size": self.input_size,
            "embedding_dim": self.embedding_dim,
            "input_name": self.input_name,
            "output_name": self.output_name,
        }

    def generate_image_embedding(self, image_bytes: bytes) -> List[float]:
        self._lazy_load()

        if self.session is None or self.input_name is None or self.output_name is None:
            raise HTTPException(status_code=500, detail="Retrieval model is not initialized")

        try:
            model_input = self._prepare_image(image_bytes)
            output = self.session.run([self.output_name], {self.input_name: model_input})[0]

            embedding = np.asarray(output[0], dtype=np.float32)
            norm = float(np.linalg.norm(embedding))
            if norm > 0:
                embedding = embedding / norm

            return embedding.tolist()
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate image embedding: {exc}",
            ) from exc


ai_model_service = AIModelService()
retrieval_embedding_service = ImageRetrievalService()

# Backward-compatible alias while the rest of the codebase is updated.
medsigclip_service = retrieval_embedding_service
