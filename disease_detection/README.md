MedsightAI service files for single-image VinBigData detection.

Files:
- `api.py`: FastAPI app with `/health` and `/predict`.
- `inference.py`: ONNX Runtime inference, TTA, WBF, and rendering.
- `export_all_to_onnx.py`: converts Sergey `stage*_fold*.pt` checkpoints to `.onnx`.

Run the exporter:

```bash
cd part_sergey
python medsight_api/export_all_to_onnx.py
```

Run the API:

```bash
cd part_sergey
python -m uvicorn medsight_api.api:app --host 0.0.0.0 --port 8000
```

Request example:

```bash
@'
import base64
import json
from pathlib import Path

image_path = Path("data/002a34c58c5b758217ed1f584ccbcfe9.png")
payload = {
    "filename": image_path.name,
    "image_base64": base64.b64encode(image_path.read_bytes()).decode("ascii"),
    "stage": 2,
    "score_thres": 0.1,
    "folds": [0, 1, 2, 3, 4],
    "tta": [0, 4],
}
Path("request.json").write_text(json.dumps(payload))
'@ | python -
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" --data-binary "@request.json"
```

The response contains detections plus `rendered_image_base64`.
