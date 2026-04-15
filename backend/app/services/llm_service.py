"""
MedGemma LLM integration service.
"""
from __future__ import annotations

import os
from importlib.util import find_spec
from io import BytesIO
from threading import Lock
from typing import Any, Dict, Iterable, Optional

import requests
from PIL import Image

from app.config.settings import settings


class MedGemmaService:
    """Lazy-loaded MedGemma image-text chat service."""

    def __init__(self) -> None:
        self._pipe = None
        self._loaded_model_id: Optional[str] = None
        self._lock = Lock()

    @staticmethod
    def _uses_bitsandbytes_quantization(model_id: str) -> bool:
        normalized = model_id.lower()
        return any(marker in normalized for marker in ("bnb", "4bit", "4-bit"))

    @staticmethod
    def _ensure_torch_set_submodule(torch_module: Any) -> None:
        if hasattr(torch_module.nn.Module, "set_submodule"):
            return

        def set_submodule(self, target: str, module: Any) -> None:
            if not target:
                raise ValueError("Cannot set an empty submodule path")

            path = target.split(".")
            parent_path = ".".join(path[:-1])
            parent = self.get_submodule(parent_path) if parent_path else self
            child_name = path[-1]
            if not hasattr(parent, child_name):
                raise AttributeError(
                    f"{parent._get_name()} has no child module named '{child_name}'"
                )
            setattr(parent, child_name, module)

        # Transformers' BitsAndBytes path expects this PyTorch API, but the
        # project's CUDA env currently uses torch 2.4 where it is absent.
        torch_module.nn.Module.set_submodule = set_submodule

    def _get_pipeline(self):
        model_id = settings.MEDGEMMA_MODEL_ID
        if self._pipe is not None and self._loaded_model_id == model_id:
            return self._pipe

        with self._lock:
            if self._pipe is not None and self._loaded_model_id == model_id:
                return self._pipe

            import torch
            from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, pipeline

            has_cuda = torch.cuda.is_available()
            requested_device = settings.MEDGEMMA_DEVICE.lower().strip()
            uses_bnb_quantization = self._uses_bitsandbytes_quantization(model_id)
            if requested_device in {"cuda", "gpu"} and not has_cuda:
                raise RuntimeError(
                    "MEDGEMMA_DEVICE is set to CUDA, but this Python environment is using CPU-only PyTorch. "
                    "Install a CUDA-enabled PyTorch build before loading MedGemma."
                )

            if requested_device == "auto":
                requested_device = "cuda" if has_cuda else "cpu"

            if requested_device in {"cuda", "gpu"}:
                gpu_index = 0
                total_vram_gb = torch.cuda.get_device_properties(gpu_index).total_memory / (1024 ** 3)
                gpu_name = torch.cuda.get_device_name(gpu_index)
                if uses_bnb_quantization:
                    missing_packages = [
                        package
                        for package in ("accelerate", "bitsandbytes")
                        if find_spec(package) is None
                    ]
                    if missing_packages:
                        raise RuntimeError(
                            f"{model_id} is a quantized BitsAndBytes model, but the backend Python environment "
                            f"is missing: {', '.join(missing_packages)}. Install them in the conda 'aic' env with "
                            "python -m pip install -U accelerate bitsandbytes."
                        )
                    self._ensure_torch_set_submodule(torch)
                elif total_vram_gb < 8:
                    raise RuntimeError(
                        f"Detected {gpu_name} with {total_vram_gb:.1f} GB VRAM. "
                        f"{model_id} in bf16 is too large for this GPU without quantization/offload. "
                        "Use a larger GPU or a quantized model such as unsloth/medgemma-1.5-4b-it-bnb-4bit."
                    )
                device = gpu_index
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            elif requested_device == "cpu":
                device = "cpu"
                torch_dtype = torch.float32
            else:
                raise RuntimeError(f"Unsupported MEDGEMMA_DEVICE value: {settings.MEDGEMMA_DEVICE}")

            token = settings.HF_TOKEN
            if token:
                os.environ["HF_TOKEN"] = token
                os.environ["HUGGING_FACE_HUB_TOKEN"] = token
            else:
                token = True
            os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "false")
            os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "1")

            pipeline_kwargs: Dict[str, Any] = {
                "model": model_id,
                "token": token,
                "dtype": torch_dtype,
            }
            if uses_bnb_quantization:
                skip_modules = [
                    "embed_tokens",
                    "embedding",
                    "lm_head",
                    "multi_modal_projector",
                    "merger",
                    "modality_projection",
                    "router",
                    "visual",
                    "vision_tower",
                    "model.embed_tokens",
                    "model.embedding",
                    "model.lm_head",
                    "model.multi_modal_projector",
                    "model.merger",
                    "model.modality_projection",
                    "model.router",
                    "model.visual",
                    "model.vision_tower",
                ]
                model_config = AutoConfig.from_pretrained(model_id, token=token)
                quantization_config = getattr(model_config, "quantization_config", None)
                if isinstance(quantization_config, dict):
                    quantization_config["llm_int8_skip_modules"] = skip_modules
                    quantization_config["bnb_4bit_compute_dtype"] = (
                        "bfloat16" if torch_dtype is torch.bfloat16 else "float16"
                    )
                elif quantization_config is not None:
                    quantization_config.llm_int8_skip_modules = skip_modules
                    quantization_config.bnb_4bit_compute_dtype = torch_dtype
                model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    config=model_config,
                    token=token,
                    dtype=torch_dtype,
                    device_map={"": device},
                )
                processor = AutoProcessor.from_pretrained(model_id, token=token)
                pipeline_kwargs["model"] = model
                pipeline_kwargs["processor"] = processor
                pipeline_kwargs.pop("token", None)
                pipeline_kwargs.pop("dtype", None)
            else:
                pipeline_kwargs["device"] = device

            self._pipe = pipeline("image-text-to-text", **pipeline_kwargs)
            self._loaded_model_id = model_id
            return self._pipe

    @staticmethod
    def _load_image(image_url: str) -> Image.Image:
        if image_url.startswith(("http://", "https://")):
            response = requests.get(
                image_url,
                headers={"User-Agent": "MedSightAI/1.0"},
                timeout=30,
            )
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")

        return Image.open(image_url).convert("RGB")

    @staticmethod
    def _format_history(conversation_history: Iterable[Any], limit: int = 8) -> str:
        items = list(conversation_history)[-limit:]
        lines: list[str] = []
        for item in items:
            if isinstance(item, dict):
                sender = item.get("sender", "user")
                message = item.get("message", "")
            else:
                sender = getattr(item, "sender", "user")
                message = getattr(item, "message", "")
            role = "Student/User" if sender == "user" else "Assistant"
            if message:
                lines.append(f"{role}: {message}")
        return "\n".join(lines) if lines else "No prior messages."

    @staticmethod
    def _format_patient_context(patient_context: Optional[Dict[str, Any]]) -> str:
        if not patient_context:
            return "No structured patient context was provided."

        allowed_keys = [
            "patientName",
            "patient_name",
            "age",
            "gender",
            "status",
            "diagnosis",
            "findings",
            "history",
            "underlying_condition",
            "mode",
        ]
        lines = []
        for key in allowed_keys:
            value = patient_context.get(key)
            if value not in (None, "", [], {}):
                lines.append(f"- {key}: {value}")
        return "\n".join(lines) if lines else "No structured patient context was provided."

    @staticmethod
    def _format_annotations(annotations: Optional[list[Dict[str, Any]]]) -> str:
        if not annotations:
            return "No current user annotations."

        lines = []
        for idx, annotation in enumerate(annotations[:12], start=1):
            x = annotation.get("x")
            y = annotation.get("y")
            width = annotation.get("width")
            height = annotation.get("height")
            label = annotation.get("label", "unlabeled")
            lines.append(
                f"{idx}. label={label}, x={x}, y={y}, width={width}, height={height}"
            )
        return "\n".join(lines)

    def _build_chat_prompt(
        self,
        *,
        conversation_history: Iterable[Any],
        student_query: str,
        mode: str,
        patient_context: Optional[Dict[str, Any]],
        current_annotations: Optional[list[Dict[str, Any]]],
        submitted_diagnosis: Optional[str],
    ) -> str:
        audience = "medical student" if mode == "student" else "clinician"
        guidance = (
            "Teach by asking guiding questions and giving concise hints. "
            "Do not reveal a definitive diagnosis unless the user explicitly asks for an explanation or feedback."
            if mode == "student"
            else "Give concise radiology assistance for the displayed image and clearly separate observations from uncertainty."
        )

        return f"""
You are MedSight AI, a careful medical imaging assistant for a {audience}.

Safety and style:
- This is educational/decision-support content, not a final clinical diagnosis.
- Base your answer on the provided chest X-ray image and case context.
- If image quality or context is insufficient, say so.
- Be specific about visible radiographic patterns, locations, and uncertainty.
- {guidance}
- Keep the answer practical and under 220 words unless the user asks for a detailed report.

Current case context:
{self._format_patient_context(patient_context)}

Current user annotations on the displayed image:
{self._format_annotations(current_annotations)}

Submitted diagnosis, if any:
{submitted_diagnosis or "None"}

Recent conversation:
{self._format_history(conversation_history)}

User message:
{student_query}
""".strip()

    @staticmethod
    def _extract_generated_text(output: Any) -> str:
        first = output[0] if isinstance(output, list) and output else output
        generated = first.get("generated_text") if isinstance(first, dict) else first

        if isinstance(generated, list) and generated:
            last = generated[-1]
            if isinstance(last, dict):
                return str(last.get("content", "")).strip()
            return str(last).strip()

        return str(generated or "").strip()

    def generate_chat_response(
        self,
        *,
        conversation_history: list,
        student_query: str,
        image_url: str,
        mode: str = "student",
        patient_context: Optional[Dict[str, Any]] = None,
        current_annotations: Optional[list[Dict[str, Any]]] = None,
        submitted_diagnosis: Optional[str] = None,
    ) -> str:
        """Generate a MedGemma response grounded in the current UI image."""
        if not image_url:
            raise ValueError("image_url is required for MedGemma image chat")

        image = self._load_image(image_url)
        prompt = self._build_chat_prompt(
            conversation_history=conversation_history,
            student_query=student_query,
            mode=mode,
            patient_context=patient_context,
            current_annotations=current_annotations,
            submitted_diagnosis=submitted_diagnosis,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        output = self._get_pipeline()(
            text=messages,
            max_new_tokens=settings.MEDGEMMA_MAX_NEW_TOKENS,
        )
        response = self._extract_generated_text(output)
        if not response:
            raise RuntimeError("MedGemma returned an empty response")
        return response

    def generate_medical_report(
        self,
        image_findings: Dict[str, Any],
        patient_history: Optional[Dict[str, Any]] = None,
        clinical_context: Optional[str] = None,
    ) -> str:
        raise NotImplementedError("Report generation is not wired to a report-specific image yet")

    def explain_prediction(
        self,
        diagnosis: str,
        findings: Dict[str, Any],
        confidence_score: float,
    ) -> str:
        raise NotImplementedError("Prediction explanation is not implemented")

    def generate_feedback(
        self,
        student_answer: Dict[str, Any],
        correct_answer: Dict[str, Any],
    ) -> str:
        raise NotImplementedError("Feedback generation should call generate_chat_response with image context")


medgemma_service = MedGemmaService()
