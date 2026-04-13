import requests
import json


NGROK_URL = "https://ad5d2f998cbe.ngrok-free.app"


def test_root():
    url = f"{NGROK_URL}/"
    print(f"Testing GET {url}")
    r = requests.get(url)
    print("Status:", r.status_code)
    print("Body:", r.text)
    print("-" * 50)


def test_generate_report():
    url = f"{NGROK_URL}/generate-report"

    payload = {
        "image_id": "5a43f10c267152bdbf23851b50c1c52d",   # ảnh có thật trong dataset VinDr
        "indication": "Shortness of breath",
        "bbox": [
          {"class_name": "Calcification", "x_min": 1567.0, "y_min": 767.0, "x_max": 1721.0, "y_max": 838.0},
          {"class_name": "Calcification", "x_min": 351.0,  "y_min": 301.0, "x_max": 927.0,  "y_max": 899.0},
          {"class_name": "Calcification", "x_min": 1653.0, "y_min": 1341.0, "x_max": 1843.0, "y_max": 1728.0},
          {"class_name": "Calcification", "x_min": 1324.0, "y_min": 540.0,  "x_max": 1803.0, "y_max": 1150.0},

          {"class_name": "Pulmonary fibrosis", "x_min": 1404.0, "y_min": 591.0, "x_max": 1818.0, "y_max": 1233.0},
          {"class_name": "Pulmonary fibrosis", "x_min": 402.0,  "y_min": 348.0, "x_max": 883.0,  "y_max": 899.0},
          {"class_name": "Pulmonary fibrosis", "x_min": 1653.0, "y_min": 1341.0, "x_max": 1843.0, "y_max": 1728.0},

          {"class_name": "Nodule/Mass", "x_min": 1404.0, "y_min": 591.0, "x_max": 1818.0, "y_max": 1233.0}
        ]
    }

    print(f"Testing POST {url}")
    r = requests.post(url, json=payload)

    print("Status:", r.status_code)

    try:
        response_json = r.json()
        print(json.dumps(response_json, indent=2, ensure_ascii=False))
    except Exception:
        print("Raw response:", r.text)

    print("-" * 50)


if __name__ == "__main__":
    test_root()
    test_generate_report()