"""
AI and retrieval model services.
"""
from __future__ import annotations

import importlib.util
import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort
from fastapi import HTTPException
from PIL import Image

from app.config.settings import BACKEND_DIR, settings


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
            raise HTTPException(status_code=500, detail=f"Retrieval model not found at {model_path}")

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
            raise HTTPException(status_code=500, detail=f"Failed to load retrieval model: {exc}") from exc

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
            raise ValueError(f"Center crop size {crop_size} is larger than resized image {(width, height)}")

        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        return image.crop((left, top, right, bottom))

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
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to preprocess retrieval image: {exc}") from exc

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
            raise HTTPException(status_code=500, detail=f"Failed to generate image embedding: {exc}") from exc


class SimilarityCamService:
    """Local SimCAM service for comparing a query case against a retrieved case."""

    def __init__(self):
        self._initialized = False
        self.device = None
        self.explainer = None
        self.transform = None
        self.image_size = 384
        self.method = "simcam"

    @staticmethod
    def _load_module(module_name: str, module_path: Path):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise HTTPException(status_code=500, detail=f"Failed to load module spec for {module_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _lazy_load(self) -> None:
        if self._initialized:
            return

        code_dir = Path(settings.SALIENCY_MODEL_CODE_PATH).expanduser().resolve()
        weights_path = Path(settings.SALIENCY_MODEL_WEIGHTS_PATH).expanduser().resolve()
        explanations_path = BACKEND_DIR / "saliency_map" / "explanations.py"

        if not code_dir.exists():
            raise HTTPException(status_code=500, detail=f"Saliency code path not found at {code_dir}")
        if not weights_path.exists():
            raise HTTPException(status_code=500, detail=f"Saliency model weights not found at {weights_path}")
        if not explanations_path.exists():
            raise HTTPException(status_code=500, detail=f"Saliency explanations file not found at {explanations_path}")

        try:
            import torch
            from torchvision import transforms

            model_module = self._load_module("saliency_model_module", code_dir / "model.py")
            explanations_module = self._load_module("saliency_explanations_module", explanations_path)

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = model_module.ConvNeXtV2(pretrained=False)
            checkpoint = torch.load(weights_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "state-dict" in checkpoint:
                checkpoint = checkpoint["state-dict"]
            model.load_state_dict(checkpoint, strict=False)
            model = model.to(self.device)
            model.eval()

            backbone = model.convnext
            target_layer = backbone.stages[3].blocks[2]
            self.explainer = explanations_module.SimCAM(model=backbone, target_layer=target_layer, fc=None)
            self.explainer = self.explainer.to(self.device)
            self.explainer.eval()

            normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            self.transform = transforms.Compose(
                [
                    transforms.Lambda(lambda image: image.convert("RGB")),
                    transforms.Resize((self.image_size, self.image_size), interpolation=_BICUBIC),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

            self._initialized = True
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to initialize saliency service: {exc}") from exc

    @staticmethod
    def _normalize_map(heatmap: np.ndarray) -> np.ndarray:
        heatmap = np.asarray(heatmap, dtype=np.float32)
        heatmap = np.clip(heatmap, 0.0, None)
        maximum = float(heatmap.max())
        minimum = float(heatmap.min())
        if maximum - minimum < 1e-8:
            return np.zeros_like(heatmap, dtype=np.float32)
        return (heatmap - minimum) / (maximum - minimum)

    @staticmethod
    def _jet_colormap(heatmap: np.ndarray) -> np.ndarray:
        x = np.clip(heatmap, 0.0, 1.0)
        red = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
        green = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
        blue = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
        return np.stack([red, green, blue], axis=-1)

    def _overlay_image(self, image: Image.Image, heatmap: np.ndarray) -> bytes:
        normalized_map = self._normalize_map(heatmap)
        resized_image = image.convert("RGB").resize((self.image_size, self.image_size), _BICUBIC)

        image_array = np.asarray(resized_image, dtype=np.float32) / 255.0
        color_map = self._jet_colormap(normalized_map)
        overlay = (0.6 * image_array) + (0.4 * color_map)
        overlay = np.clip(overlay * 255.0, 0.0, 255.0).astype(np.uint8)

        buffer = io.BytesIO()
        Image.fromarray(overlay).save(buffer, format="PNG")
        return buffer.getvalue()

    def _prepare_tensor(self, image_bytes: bytes):
        self._lazy_load()
        if self.transform is None or self.device is None:
            raise HTTPException(status_code=500, detail="Saliency service is not initialized")

        try:
            import torch

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            return image, tensor.to(dtype=torch.float32)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to preprocess saliency image: {exc}") from exc

    def generate_pair_saliency(self, query_image_bytes: bytes, similar_image_bytes: bytes) -> Dict[str, Any]:
        self._lazy_load()
        if self.explainer is None:
            raise HTTPException(status_code=500, detail="Saliency service is not initialized")

        query_image, query_tensor = self._prepare_tensor(query_image_bytes)
        similar_image, similar_tensor = self._prepare_tensor(similar_image_bytes)

        try:
            saliency_maps = self.explainer(query_tensor, similar_tensor)
            saliency_maps = saliency_maps.detach().cpu().numpy()

            if saliency_maps.ndim != 4 or saliency_maps.shape[0] < 1 or saliency_maps.shape[1] < 2:
                raise HTTPException(status_code=500, detail=f"Unexpected saliency output shape: {saliency_maps.shape}")

            query_overlay = self._overlay_image(query_image, saliency_maps[0, 0])
            similar_overlay = self._overlay_image(similar_image, saliency_maps[0, 1])

            return {
                "method": self.method,
                "image_size": self.image_size,
                "query_overlay_bytes": query_overlay,
                "similar_overlay_bytes": similar_overlay,
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to generate saliency map: {exc}") from exc


ai_model_service = AIModelService()
retrieval_embedding_service = ImageRetrievalService()
similarity_cam_service = SimilarityCamService()

# Backward-compatible alias while the rest of the codebase is updated.
medsigclip_service = retrieval_embedding_service
