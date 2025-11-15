from generate_report import generate_clinical_report_from_path
import os
import traceback

patient_metadata = {
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
    "disease_type": "Healthy",
    "indication": "Evaluation of chest symptoms.",
    "comparison_info": "None",
}

image_path = os.path.join("Images", "h0001.png")

print(">>> Starting test_local.py")
print(">>> Image path:", image_path)

try:
    result = generate_clinical_report_from_path(image_path, patient_metadata)
    print("===== REPORT RESULT =====")
    print("patient_metadata:", result["patient_metadata"])
    print("radiology_report:", result["radiology_report"])
except Exception:
    print(">>> ERROR OCCURRED:")
    traceback.print_exc()
