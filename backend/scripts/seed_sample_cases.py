"""
Seed patient and case records through the public API using local sample images.

Default behavior is a one-record smoke test. Use --limit to scale up after
verifying the backend, database, and S3 flow are working correctly.
"""
from __future__ import annotations

import argparse
import mimetypes
import random
from pathlib import Path
from typing import Dict, Iterable, List

import requests
from requests import Response


DATASET_FOLDERS: Dict[str, Path] = {
    "COVID-19": Path(r"C:\Users\HP\Coding\MedSightAI-DataForLife\covidx_sample_300\COVID-19"),
    "normal": Path(r"C:\Users\HP\Coding\MedSightAI-DataForLife\covidx_sample_300\normal"),
    "pneumonia": Path(r"C:\Users\HP\Coding\MedSightAI-DataForLife\covidx_sample_300\pneumonia"),
}

LAST_NAMES = [
    "Nguyen", "Tran", "Le", "Pham", "Hoang", "Huynh", "Phan", "Vu", "Vo",
    "Dang", "Do", "Ngo", "Ho", "Duong", "Ly", "Bui", "Dinh", "Truong",
]
MIDDLE_NAMES = [
    "Van", "Thi", "Huu", "Minh", "Gia", "Ngoc", "Quoc", "Thanh", "Hoai",
    "Khanh", "Bao", "Anh", "Tuan", "Nhat", "Duc", "Thien", "Dieu",
]
FIRST_NAMES = [
    "An", "Binh", "Dung", "Huy", "Khang", "Long", "Minh", "Nam", "Phuc",
    "Quan", "Son", "Trung", "Tuan", "Vy", "Lan", "Ha", "Nhu", "Ngoc",
    "Hanh", "Linh", "Thao", "Quynh", "Tram",
]
GENDERS = ["Nam", "Nu"]
BLOOD_TYPES = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]
PATIENT_STATUSES = ["stable", "improving", "critical"]
PHONE_PREFIXES = ["+8432", "+8433", "+8434", "+8435", "+8436", "+8437", "+8438", "+8439", "+8470", "+8476", "+8477", "+8478", "+8479", "+8481", "+8482", "+8483", "+8484", "+8485", "+8488", "+8489", "+8490", "+8491", "+8493", "+8494", "+8496", "+8497", "+8498"]

CASE_METADATA = {
    "COVID-19": {
        "diagnosis": "COVID-19",
        "findings": "Chest X-ray sample labeled COVID-19. Bilateral pulmonary opacities should be correlated clinically.",
    },
    "pneumonia": {
        "diagnosis": "Pneumonia",
        "findings": "Chest X-ray sample labeled pneumonia. Patchy or focal air-space opacity is expected in this class.",
    },
    "normal": {
        "diagnosis": "Normal",
        "findings": "Chest X-ray sample labeled normal. No acute cardiopulmonary abnormality indicated by the dataset label.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed Vietnamese patients and image cases via the backend API.")
    parser.add_argument("--base-url", default="http://localhost:8000/api/v1", help="Backend API base URL.")
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of images/patients to import. Default is 1 for a safe smoke test.",
    )
    parser.add_argument(
        "--class-filter",
        choices=["COVID-19", "normal", "pneumonia", "all"],
        default="all",
        help="Restrict import to one label folder or use all folders.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP request timeout in seconds.",
    )
    return parser.parse_args()


def generate_vietnamese_name() -> str:
    return f"{random.choice(LAST_NAMES)} {random.choice(MIDDLE_NAMES)} {random.choice(FIRST_NAMES)}"


def generate_phone_number() -> str:
    return random.choice(PHONE_PREFIXES) + "".join(str(random.randint(0, 9)) for _ in range(7))


def build_patient_payload() -> Dict[str, object]:
    return {
        "name": generate_vietnamese_name(),
        "age": random.randint(18, 85),
        "gender": random.choice(GENDERS),
        "history": None,
        "blood_type": random.choice(BLOOD_TYPES),
        "status": random.choice(PATIENT_STATUSES),
        "underlying_condition": {
            "hypertension": random.choice([True, False]),
            "diabetes": random.choice([True, False]),
            "asthma": random.choice([True, False]),
        },
        "phone_number": generate_phone_number(),
        "fcm_token": None,
    }


def collect_images(class_filter: str) -> List[Dict[str, str]]:
    class_names = DATASET_FOLDERS.keys() if class_filter == "all" else [class_filter]
    images: List[Dict[str, str]] = []

    for class_name in class_names:
        folder = DATASET_FOLDERS[class_name]
        if not folder.exists():
            raise FileNotFoundError(f"Dataset folder not found: {folder}")

        for image_path in sorted(path for path in folder.iterdir() if path.is_file()):
            images.append({"label": class_name, "path": str(image_path)})

    return images


def iter_selected_images(images: List[Dict[str, str]], limit: int) -> Iterable[Dict[str, str]]:
    if limit <= 0:
        return []
    return images[:limit]


def ensure_success(response: Response, action: str) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"{action} failed with {response.status_code}: {response.text}") from exc


def check_backend(base_url: str, timeout: int) -> None:
    health_url = base_url.removesuffix("/api/v1") + "/"
    try:
        response = requests.get(health_url, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Backend is not reachable at {health_url}. Start the FastAPI server before running this script."
        ) from exc
    ensure_success(response, "Backend health check")


def create_patient(session: requests.Session, base_url: str, timeout: int) -> Dict[str, object]:
    response = session.post(
        f"{base_url}/patients/",
        json=build_patient_payload(),
        timeout=timeout,
    )
    ensure_success(response, "Create patient")
    return response.json()


def upload_case_image(
    session: requests.Session,
    base_url: str,
    patient_id: str,
    image_path: Path,
    timeout: int,
) -> Dict[str, object]:
    content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    with image_path.open("rb") as image_file:
        response = session.post(
            f"{base_url}/cases/upload",
            params={"patient_id": patient_id},
            files={"file": (image_path.name, image_file, content_type)},
            timeout=timeout,
        )
    ensure_success(response, f"Upload case image {image_path.name}")
    return response.json()


def update_case_metadata(
    session: requests.Session,
    base_url: str,
    case_id: str,
    label: str,
    timeout: int,
) -> Dict[str, object]:
    response = session.put(
        f"{base_url}/cases/{case_id}",
        json=CASE_METADATA[label],
        timeout=timeout,
    )
    ensure_success(response, f"Update case metadata for {case_id}")
    return response.json()


def main() -> None:
    args = parse_args()
    images = collect_images(args.class_filter)

    if not images:
        raise RuntimeError("No images found for the requested filter.")

    selected_images = list(iter_selected_images(images, args.limit))
    if not selected_images:
        raise RuntimeError("No images selected. Increase --limit.")

    print(f"Found {len(images)} images. Preparing to import {len(selected_images)} record(s).")

    session = requests.Session()
    check_backend(args.base_url, args.timeout)

    success_count = 0
    for index, item in enumerate(selected_images, start=1):
        label = item["label"]
        image_path = Path(item["path"])
        print(f"[{index}/{len(selected_images)}] Creating patient and case for {label}: {image_path.name}")

        patient = create_patient(session, args.base_url, args.timeout)
        case = upload_case_image(session, args.base_url, str(patient["id"]), image_path, args.timeout)
        updated_case = update_case_metadata(session, args.base_url, str(case["id"]), label, args.timeout)

        print(
            f"  Patient {patient['id']} | {patient['name']} | Case {updated_case['id']} | "
            f"Diagnosis: {updated_case.get('diagnosis')}"
        )
        success_count += 1

    print(f"Completed. Imported {success_count} patient/case record(s).")


if __name__ == "__main__":
    main()
