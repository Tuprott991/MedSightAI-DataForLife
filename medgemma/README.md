````markdown
# MedGemma 4B-IT – Chest X-ray Report Demo

1. Tạo môi trường Python riêng (conda).
2. Cài PyTorch + các thư viện cần thiết.
3. Cấu hình Hugging Face token qua `.env`.
4. Chạy thử `test_local.py` để sinh báo cáo.

---

## 1. Yêu cầu hệ thống

- Python **3.9–3.11** (khuyến nghị 3.10)
- `conda` (Anaconda / Miniconda) – khuyến nghị
- GPU NVIDIA + driver + CUDA driver (nếu muốn dùng GPU)
- Tài khoản Hugging Face + token có quyền truy cập model `google/medgemma-4b-it`

---

## 2. Tạo môi trường conda

```bash
conda create -n medgemma python=3.10 -y
conda activate medgemma
```

---

## 3. Cài PyTorch (GPU hoặc CPU)

### 3.1. Nếu có GPU NVIDIA (khuyến nghị)

Vào trang PyTorch để chọn lệnh phù hợp CUDA:
[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Ví dụ với CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3.2. Nếu chỉ muốn chạy CPU (rất chậm, không khuyến nghị cho 4B)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## 4. Cài các thư viện Python cần thiết

Trong env `medgemma`:

```bash
pip install --upgrade \
    transformers \
    accelerate \
    bitsandbytes \
    huggingface_hub \
    pillow \
    python-dotenv
```

Nếu một số option shell bị lỗi trên Windows, có thể cài từng dòng:

```bash
pip install --upgrade transformers
pip install --upgrade accelerate
pip install --upgrade bitsandbytes
pip install --upgrade huggingface_hub
pip install pillow
pip install python-dotenv
```

---

## 5. Cấu hình Hugging Face token (`.env`)

Trong thư mục `medgemma` (chung cấp với `generate_report.py`, `test_local.py`), tạo file `.env`:

```text
HF_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Trong đó `xxxxxxxx...` là token lấy từ trang Hugging Face:
Profile → Settings → Access Tokens.

---

## 6. Chuẩn bị dữ liệu test

Mặc định `test_local.py` đang dùng một ảnh demo:

```python
image_path = os.path.join("Images", "h0001.png")
```

Hãy đảm bảo:

* Thư mục: `medgemma/Images/`
* Có file: `h0001.png` (ảnh X-quang ngực)

Ví dụ cấu trúc:

```text
medgemma/
├─ generate_report.py
├─ test_local.py
├─ Images/
│  └─ h0001.png
├─ .env
└─ ...
```

---

## 7. Chạy test local

Trong thư mục `medgemma`:

```bash
conda activate medgemma
python test_local.py
```

Nếu mọi thứ OK, bạn sẽ thấy log tương tự:

```text
>>> Starting test_local.py
>>> Image path: Images\h0001.png
>>> Calling get_pipe()...
>>> torch version: ...
>>> CUDA available: True
>>> GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU
>>> MedGemma pipeline is ready.
>>> Building prompt...
>>> Running inference...
>>> Inference done, parsing output...
>>> Report generation finished.
===== REPORT RESULT =====
patient_metadata: {...}
radiology_report: {...}
```

---

## 8. Cấu trúc kết quả trả về

Hàm `generate_clinical_report_from_path` trong `generate_report.py` trả về một dict dạng:

```python
{
    "patient_metadata": {
        "patient_id": "P0001",
        "age": 34,
        "sex": "F",
        "study_id": "S-P0001-2025-11-01",
        "image_filename": "h0001.png",
        "image_type": "PA",
        "views": "PA",
        "image_height": 2048,
        "image_width": 2048,
        "source": "test",
        "bbox": "none",
        "target": "no",
        "disease_type": "healthy",
        "indication": "Evaluation of chest symptoms.",
        "comparison_info": "None",
    },
    "radiology_report": {
        "MeSH": "Normal",
        "Problems": "healthy",
        "Image": "X-ray Chest PA",
        "Indication": "Evaluation of chest symptoms.",
        "Comparison": "None",
        "Findings": "The cardiomediastinal silhouette is within normal limits. ...",
        "Impression": "Normal chest radiograph.",
        "raw_report_text": "==============================\nPATIENT INFORMATION\n..."
    }
}
```

