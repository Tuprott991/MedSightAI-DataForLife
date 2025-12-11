from fastapi import APIRouter, UploadFile, File
import tempfile
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from service import load_csr_model, preprocess_image, infer_cams

router = APIRouter()

CHECKPOINT_PATH = "/home/aaronpham5504/Coding/SoftAI---DataForLife---MedSightAI/model_inference/csr_phase1.pth"  # Sửa lại đường dẫn checkpoint
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly',
    'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity',
    'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening',
    'Pneumothorax', 'Pulmonary fibrosis'
]

model = load_csr_model(CHECKPOINT_PATH, DEVICE)

@router.post("/cam-inference/")
async def cam_inference(file: UploadFile = File(...), threshold: float = 0.5):
    # Lưu file tạm
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Tiền xử lý ảnh
    image_tensor = preprocess_image(tmp_path)
    # Inference
    probs, cams = infer_cams(model, image_tensor, DEVICE)
    

    # Lọc các class có prob >= threshold và lấy bbox từ attention map (CAM)
    top_indices = probs.argsort()[::-1][:14]
    filtered = []
    for idx in top_indices:
        if probs[idx] >= threshold:
            cam_np = np.array(cams[idx])
            filtered.append({
                "class_idx": int(idx),
                "prob": float(probs[idx]),
                "cam": cam_np.tolist(),
            })

    # Nếu không có class nào vượt threshold, trả về rỗng
    if not filtered:
        return {"top_classes": [], "cams": [], "bboxes": []}

    # Trả về các class, CAMs và bbox vượt threshold
    results =  {
        "top_classes": [{"class_idx": item["class_idx"], "prob": item["prob"], "concepts": CLASS_NAMES[item["class_idx"]]} for item in filtered],
        "cams": [item["cam"] for item in filtered],
    }

    bboxes = visualize_attention_weights_on_image(tmp_path, results["cams"], threshold=0.8)
    results["bboxes"] = bboxes
    import os; os.remove(tmp_path)

    return results

def visualize_attention_weights_on_image(image_path, cams, alpha=0.4, threshold=0.9):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Không đọc được ảnh.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes = []
    for cam in cams:
        # plt.figure(figsize=(8, 8))
        # plt.imshow(img)
        cam_np = np.array(cam)
        cam_resized = cv2.resize(cam_np, (img.shape[1], img.shape[0]))
        cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
        mask = (cam_norm > threshold).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bbox = None
        if contours:
            all_points = np.concatenate(contours, axis=0)
            x, y, w, h = cv2.boundingRect(all_points)
            bbox = [int(x), int(y), int(x + w), int(y + h)]
            # plt.gca().add_patch(
            #     plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
            # )
        bboxes.append(bbox)
        # Nếu muốn overlay CAM cho từng concept thì bỏ comment dòng dưới
        # plt.imshow(cam_norm * mask, cmap='jet', alpha=alpha)
        # plt.axis('off')
        # plt.show()
    return bboxes