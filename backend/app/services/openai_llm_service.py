"""
OpenAI GPT-backed medical vision/text service.
"""
from __future__ import annotations

import base64
import json
import logging
import mimetypes
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, Iterator, Optional
from urllib.parse import urlparse

from app.config.settings import settings

logger = logging.getLogger(__name__)


DOCTOR_CHAT_SYSTEM_PROMPT = """
You are MedSight AI Doctor Assist, a careful chest imaging assistant for licensed clinicians.

Rules:
- Use the provided image, patient context, and recent chat history.
- Separate visible imaging observations from interpretation and uncertainty.
- Do not invent findings, measurements, comparison studies, or clinical facts.
- Give decision-support, not a final clinical diagnosis.
- Recommend escalation or additional imaging/labs only when supported by the context.
- Keep answers concise and clinically practical unless the doctor asks for more detail.
""".strip()


STUDENT_CHAT_SYSTEM_PROMPT = """
You are MedSight AI Tutor, a chest X-ray teaching assistant for medical students.

Rules:
- Use the provided image, student annotations, submitted diagnosis, and case context.
- Teach with guiding questions, short hints, and concrete visual landmarks.
- Do not reveal the final diagnosis immediately unless the student submitted an answer or explicitly asks for feedback/explanation.
- When annotations are present, discuss likely location quality, missed regions, and next observation steps.
- Be encouraging but precise; do not overstate certainty from the image alone.
- Keep answers educational and under 220 words unless asked for a detailed explanation.
""".strip()


REPORT_GENERATION_SYSTEM_PROMPT = """
You are MedSight AI Report Writer, a radiology report drafting assistant for chest X-ray cases.

Rules:
- Generate a structured draft report from the image, patient context, case context, and AI detections.
- Do not fabricate unavailable history, prior comparisons, measurements, or certainty.
- If image quality or clinical context is limited, state the limitation in the findings or impression.
- Use Vietnamese clinical prose for report values because the current report UI is Vietnamese.
- Return only JSON matching the requested schema.
- This is a draft for doctor review, not a final signed diagnosis.
""".strip()


REPORT_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["thong_tin_benh_nhan", "bao_cao_x_quang", "safety_note"],
    "properties": {
        "thong_tin_benh_nhan": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "ho_ten",
                "tuoi",
                "gioi_tinh",
                "nhom_mau",
                "ngay_chup",
                "ngay_doc_phim",
                "chan_doan_lam_sang",
                "bac_si_doc_phim",
            ],
            "properties": {
                "ho_ten": {"type": "string"},
                "tuoi": {"type": "string"},
                "gioi_tinh": {"type": "string"},
                "nhom_mau": {"type": "string"},
                "ngay_chup": {"type": "string"},
                "ngay_doc_phim": {"type": "string"},
                "chan_doan_lam_sang": {"type": "string"},
                "bac_si_doc_phim": {"type": "string"},
            },
        },
        "bao_cao_x_quang": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "MeSH",
                "loai_anh",
                "chi_dinh",
                "so_sanh",
                "mo_ta",
                "ket_luan",
            ],
            "properties": {
                "MeSH": {"type": "string"},
                "loai_anh": {"type": "string"},
                "chi_dinh": {"type": "string"},
                "so_sanh": {"type": "string"},
                "mo_ta": {"type": "string"},
                "ket_luan": {"type": "string"},
            },
        },
        "safety_note": {"type": "string"},
    },
}


class OpenAILLMService:
    """GPT image-grounded chat and report generation service."""

    def __init__(self) -> None:
        self._client = None
        self._lock = Lock()

    def _get_client(self):
        if self._client is not None:
            return self._client

        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        with self._lock:
            if self._client is not None:
                return self._client

            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "The installed openai package is too old for the GPT service. "
                    "Install openai>=1.75.0 in the backend Python environment."
                ) from exc

            self._client = OpenAI(
                api_key=settings.OPENAI_API_KEY,
                timeout=settings.OPENAI_TIMEOUT_SECONDS,
                max_retries=settings.OPENAI_MAX_RETRIES,
            )
            return self._client

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
            role = "User" if sender == "user" else "Assistant"
            if message:
                lines.append(f"{role}: {message}")
        return "\n".join(lines) if lines else "No prior messages."

    @staticmethod
    def _format_mapping(data: Optional[Dict[str, Any]]) -> str:
        if not data:
            return "No structured context was provided."

        lines = []
        for key, value in data.items():
            if value in (None, "", [], {}):
                continue
            if isinstance(value, (dict, list)):
                value_text = json.dumps(value, ensure_ascii=False, default=str)
            else:
                value_text = str(value)
            lines.append(f"- {key}: {value_text}")
        return "\n".join(lines) if lines else "No structured context was provided."

    @staticmethod
    def _format_annotations(annotations: Optional[list[Dict[str, Any]]]) -> str:
        if not annotations:
            return "No current user annotations."

        lines = []
        for idx, annotation in enumerate(annotations[:20], start=1):
            label = annotation.get("label") or annotation.get("type") or "unlabeled"
            x = annotation.get("x")
            y = annotation.get("y")
            width = annotation.get("width")
            height = annotation.get("height")
            extra = {
                key: value
                for key, value in annotation.items()
                if key not in {"label", "type", "x", "y", "width", "height"}
            }
            extra_text = f", extra={json.dumps(extra, ensure_ascii=False)}" if extra else ""
            lines.append(
                f"{idx}. label={label}, x={x}, y={y}, width={width}, height={height}{extra_text}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_detections(detections: Optional[list[Dict[str, Any]]]) -> str:
        if not detections:
            return "No AI detections were provided."

        lines = []
        for idx, detection in enumerate(detections[:30], start=1):
            lines.append(f"{idx}. {json.dumps(detection, ensure_ascii=False, default=str)}")
        return "\n".join(lines)

    @staticmethod
    def _build_public_s3_url(s3_key: str) -> str:
        clean_key = s3_key.lstrip("/").replace("\\", "/")
        if settings.AWS_REGION == "us-east-1":
            return f"https://{settings.S3_BUCKET_NAME}.s3.amazonaws.com/{clean_key}"
        return f"https://{settings.S3_BUCKET_NAME}.s3.{settings.AWS_REGION}.amazonaws.com/{clean_key}"

    @classmethod
    def _normalize_image_url(cls, image_url: str) -> str:
        if not image_url:
            raise ValueError("image_url is required")

        parsed = urlparse(image_url)
        if parsed.scheme in {"http", "https"} or image_url.startswith("data:image/"):
            return image_url

        path = Path(image_url)
        if path.exists() and path.is_file():
            mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime_type};base64,{encoded}"

        if not parsed.scheme:
            return cls._build_public_s3_url(image_url)

        raise ValueError("image_url must be an HTTP(S) URL, data URL, S3 key, or local file path")

    @staticmethod
    def _extract_output_text(response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text).strip()

        chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(str(text))
        return "\n".join(chunks).strip()

    def _build_response_kwargs(
        self,
        *,
        instructions: str,
        prompt: str,
        image_url: str,
        max_output_tokens: int,
        metadata: Optional[Dict[str, str]] = None,
        text_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized_image_url = self._normalize_image_url(image_url)
        kwargs: Dict[str, Any] = {
            "model": settings.OPENAI_LLM_MODEL,
            "instructions": instructions,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": normalized_image_url,
                            "detail": "high",
                        },
                        {"type": "input_text", "text": prompt},
                    ],
                }
            ],
            "max_output_tokens": max_output_tokens,
            "store": False,
        }
        if metadata:
            kwargs["metadata"] = metadata
        if text_format:
            kwargs["text"] = {"format": text_format}
        return kwargs

    def _create_response(
        self,
        *,
        instructions: str,
        prompt: str,
        image_url: str,
        max_output_tokens: int,
        metadata: Optional[Dict[str, str]] = None,
        text_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        client = self._get_client()
        kwargs = self._build_response_kwargs(
            instructions=instructions,
            prompt=prompt,
            image_url=image_url,
            max_output_tokens=max_output_tokens,
            metadata=metadata,
            text_format=text_format,
        )

        logger.info(
            "[OPENAI-LLM] Calling OpenAI model=%s task=%s",
            settings.OPENAI_LLM_MODEL,
            (metadata or {}).get("task", "unknown"),
        )
        response = client.responses.create(**kwargs)
        output_text = self._extract_output_text(response)
        if not output_text:
            raise RuntimeError("OpenAI returned an empty response")
        return output_text

    @staticmethod
    def _chat_instructions(mode: str) -> str:
        return DOCTOR_CHAT_SYSTEM_PROMPT if mode == "doctor" else STUDENT_CHAT_SYSTEM_PROMPT

    def _build_chat_prompt(
        self,
        *,
        conversation_history: list,
        student_query: str,
        patient_context: Optional[Dict[str, Any]],
        current_annotations: Optional[list[Dict[str, Any]]],
        submitted_diagnosis: Optional[str],
    ) -> str:
        return f"""
Current case context:
{self._format_mapping(patient_context)}

Current student/user annotations on the displayed image:
{self._format_annotations(current_annotations)}

Submitted diagnosis, if any:
{submitted_diagnosis or "None"}

Recent conversation:
{self._format_history(conversation_history)}

User message:
{student_query}
""".strip()

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
        """Generate a GPT response grounded in the current UI image."""
        mode = "doctor" if mode == "doctor" else "student"

        return self._create_response(
            instructions=self._chat_instructions(mode),
            prompt=self._build_chat_prompt(
                conversation_history=conversation_history,
                student_query=student_query,
                patient_context=patient_context,
                current_annotations=current_annotations,
                submitted_diagnosis=submitted_diagnosis,
            ),
            image_url=image_url,
            max_output_tokens=settings.OPENAI_CHAT_MAX_OUTPUT_TOKENS,
            metadata={"task": f"{mode}_chat"},
        )

    def stream_chat_response(
        self,
        *,
        conversation_history: list,
        student_query: str,
        image_url: str,
        mode: str = "student",
        patient_context: Optional[Dict[str, Any]] = None,
        current_annotations: Optional[list[Dict[str, Any]]] = None,
        submitted_diagnosis: Optional[str] = None,
    ) -> Iterator[str]:
        """Yield GPT response text deltas grounded in the current UI image."""
        mode = "doctor" if mode == "doctor" else "student"
        client = self._get_client()
        kwargs = self._build_response_kwargs(
            instructions=self._chat_instructions(mode),
            prompt=self._build_chat_prompt(
                conversation_history=conversation_history,
                student_query=student_query,
                patient_context=patient_context,
                current_annotations=current_annotations,
                submitted_diagnosis=submitted_diagnosis,
            ),
            image_url=image_url,
            max_output_tokens=settings.OPENAI_CHAT_MAX_OUTPUT_TOKENS,
            metadata={"task": f"{mode}_chat_stream"},
        )

        logger.info(
            "[OPENAI-LLM] Streaming OpenAI model=%s task=%s",
            settings.OPENAI_LLM_MODEL,
            kwargs.get("metadata", {}).get("task", "unknown"),
        )
        stream = client.responses.create(**kwargs, stream=True)
        yielded_delta = False
        fallback_done_text: Optional[str] = None

        for event in stream:
            event_type = getattr(event, "type", "")
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    yielded_delta = True
                    yield str(delta)
            elif event_type == "response.output_text.done":
                text = getattr(event, "text", None)
                if text and not yielded_delta:
                    fallback_done_text = str(text)
            elif event_type in {"response.failed", "response.incomplete", "error"}:
                error = getattr(event, "error", None)
                message = getattr(error, "message", None) if error else None
                raise RuntimeError(message or f"OpenAI stream failed: {event_type}")

        if fallback_done_text:
            yield fallback_done_text

    def generate_medical_report(
        self,
        *,
        image_url: str,
        patient_context: Dict[str, Any],
        case_context: Dict[str, Any],
        ai_context: Optional[Dict[str, Any]] = None,
        detections: Optional[list[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Generate a structured report draft for the report UI."""
        prompt = f"""
Create a chest X-ray report draft in JSON.

Patient context:
{self._format_mapping(patient_context)}

Case context:
{self._format_mapping(case_context)}

AI analysis context:
{self._format_mapping(ai_context)}

AI detections and bounding boxes:
{self._format_detections(detections)}

Field guidance:
- thong_tin_benh_nhan.ho_ten: patient name or "N/A".
- thong_tin_benh_nhan.tuoi: age as text or "N/A".
- thong_tin_benh_nhan.gioi_tinh: gender or "N/A".
- thong_tin_benh_nhan.nhom_mau: blood type or "N/A".
- thong_tin_benh_nhan.ngay_chup: case image date or "N/A".
- thong_tin_benh_nhan.ngay_doc_phim: today's report generation date if no read date is supplied.
- thong_tin_benh_nhan.chan_doan_lam_sang: clinical diagnosis/context or "N/A".
- thong_tin_benh_nhan.bac_si_doc_phim: use "AI draft - doctor review required".
- bao_cao_x_quang.MeSH: comma-separated key imaging concepts.
- bao_cao_x_quang.loai_anh: image type/projection.
- bao_cao_x_quang.chi_dinh: clinical indication.
- bao_cao_x_quang.so_sanh: comparison information, or state that no prior study is available.
- bao_cao_x_quang.mo_ta: detailed findings.
- bao_cao_x_quang.ket_luan: concise impression and review recommendation.
""".strip()

        output_text = self._create_response(
            instructions=REPORT_GENERATION_SYSTEM_PROMPT,
            prompt=prompt,
            image_url=image_url,
            max_output_tokens=settings.OPENAI_REPORT_MAX_OUTPUT_TOKENS,
            metadata={"task": "report_generation"},
            text_format={
                "type": "json_schema",
                "name": "medsight_radiology_report",
                "schema": REPORT_JSON_SCHEMA,
                "strict": True,
            },
        )

        try:
            report = json.loads(output_text)
        except json.JSONDecodeError as exc:
            logger.error("[OPENAI-LLM] Report JSON parse failed", exc_info=True)
            raise RuntimeError("OpenAI report response was not valid JSON") from exc

        if not isinstance(report, dict):
            raise RuntimeError("OpenAI report response must be a JSON object")
        return report


openai_llm_service = OpenAILLMService()
