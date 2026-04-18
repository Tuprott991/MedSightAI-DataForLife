import base64
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .inference import DEFAULT_FOLDS, DEFAULT_TTA, encode_image_b64, load_image_from_bytes, predict_image


app = FastAPI(title="MedsightAI VinBigData Detection API", version="1.0.0")


class PredictRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded input image bytes")
    filename: str = Field(default="image.jpg")
    stage: int = Field(default=2)
    img_size: int = Field(default=640)
    wbf_iou: float = Field(default=0.4)
    score_thres: float = Field(default=0.1)
    device: str = Field(default="")
    folds: List[int] = Field(default_factory=lambda: DEFAULT_FOLDS.copy())
    tta: List[int] = Field(default_factory=lambda: DEFAULT_TTA.copy())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        image_bytes = base64.b64decode(request.image_base64)
        image_bgr = load_image_from_bytes(image_bytes)

        result = predict_image(
            image_bgr=image_bgr,
            stage=request.stage,
            folds=request.folds,
            tta=request.tta,
            img_size=request.img_size,
            wbf_iou=request.wbf_iou,
            score_thres=request.score_thres,
            device=request.device,
        )
        rendered = result.pop("rendered_image_bgr")
        result["rendered_image_base64"] = encode_image_b64(rendered, request.filename)
        result["filename"] = request.filename
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
