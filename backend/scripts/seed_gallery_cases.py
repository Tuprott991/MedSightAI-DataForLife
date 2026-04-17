"""
Seed a flat COVIDx gallery into Neon and Zilliz, one image at a time.

The script creates one patient and one case per image through the public API,
then immediately calls the embedding endpoint so the gallery is searchable.
"""

from __future__ import annotations

import argparse
import hashlib
import mimetypes
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Set
import requests
from requests import Response


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "covidx_sample_300"
DEFAULT_LABELS_FILE = REPO_ROOT / "backend" / "test.txt"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".dicom", ".dcm"}
SHA256_PATTERN = re.compile(r"SHA256:([0-9a-f]{64})", re.IGNORECASE)
LABEL_ALIASES = {
    "normal": "Normal",
    "pneumonia": "pneumonia",
    "covid-19": "COVID19",
    "covid19": "COVID19",
}

LAST_NAMES = [
    "Nguyen",
    "Tran",
    "Le",
    "Pham",
    "Hoang",
    "Huynh",
    "Phan",
    "Vu",
    "Vo",
    "Dang",
    "Do",
    "Ngo",
    "Ho",
    "Duong",
    "Ly",
    "Bui",
    "Dinh",
    "Truong",
]
MIDDLE_NAMES = [
    "Van",
    "Thi",
    "Huu",
    "Minh",
    "Gia",
    "Ngoc",
    "Quoc",
    "Thanh",
    "Hoai",
    "Khanh",
    "Bao",
    "Anh",
    "Tuan",
    "Nhat",
    "Duc",
    "Thien",
    "Dieu",
]
FIRST_NAMES = [
    "An",
    "Binh",
    "Dung",
    "Huy",
    "Khang",
    "Long",
    "Minh",
    "Nam",
    "Phuc",
    "Quan",
    "Son",
    "Trung",
    "Tuan",
    "Vy",
    "Lan",
    "Ha",
    "Nhu",
    "Ngoc",
    "Hanh",
    "Linh",
    "Thao",
    "Quynh",
    "Tram",
]
GENDERS = ["Nam", "Nu"]
BLOOD_TYPES = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]
PATIENT_STATUSES = ["stable", "improving", "critical"]
PHONE_PREFIXES = [
    "+8432",
    "+8433",
    "+8434",
    "+8435",
    "+8436",
    "+8437",
    "+8438",
    "+8439",
    "+8470",
    "+8476",
    "+8477",
    "+8478",
    "+8479",
    "+8481",
    "+8482",
    "+8483",
    "+8484",
    "+8485",
    "+8488",
    "+8489",
    "+8490",
    "+8491",
    "+8493",
    "+8494",
    "+8496",
    "+8497",
    "+8498",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed a searchable COVIDx gallery through the backend API."
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/api/v1",
        help="Backend API base URL.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Folder containing the gallery images.",
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=DEFAULT_LABELS_FILE,
        help="Text file mapping each image filename to a doctor-provided diagnosis.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Number of images to import. Default: 300.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N images. Useful for resuming partial imports.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--findings",
        default="Imported into the COVIDx retrieval gallery. Doctor diagnosis already available.",
        help="Findings metadata stored on each case.",
    )
    parser.add_argument(
        "--embed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Call /similarity/embed after each upload.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue importing remaining images if one record fails.",
    )
    return parser.parse_args()


def generate_vietnamese_name() -> str:
    return f"{random.choice(LAST_NAMES)} {random.choice(MIDDLE_NAMES)} {random.choice(FIRST_NAMES)}"


def generate_phone_number() -> str:
    return random.choice(PHONE_PREFIXES) + "".join(
        str(random.randint(0, 9)) for _ in range(7)
    )


def build_patient_payload(diagnosis: str, findings: str) -> Dict[str, object]:
    history_date = datetime.now().strftime("%m-%d-%Y")
    return {
        "name": generate_vietnamese_name(),
        "age": random.randint(18, 85),
        "gender": random.choice(GENDERS),
        "history": {
            history_date: {
                "diagnosis": diagnosis,
                "findings": findings,
            }
        },
        "blood_type": random.choice(BLOOD_TYPES),
        "status": "Processed",
        "underlying_condition": {
            "hypertension": random.choice([True, False]),
            "diabetes": random.choice([True, False]),
            "asthma": random.choice([True, False]),
        },
        "phone_number": generate_phone_number(),
        "fcm_token": None,
    }


def collect_images(dataset_root: Path) -> List[Path]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

    images = [
        path
        for path in sorted(dataset_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return images


def normalize_diagnosis(raw_label: str) -> str:
    normalized = raw_label.strip().lower()
    if normalized not in LABEL_ALIASES:
        raise ValueError(f"Unsupported diagnosis label in test.txt: {raw_label}")
    return LABEL_ALIASES[normalized]


def load_label_map(labels_file: Path) -> Dict[str, str]:
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")

    label_map: Dict[str, str] = {}
    for line_number, raw_line in enumerate(labels_file.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Invalid label line {line_number}: {raw_line}")

        filename = parts[1]
        diagnosis = normalize_diagnosis(parts[2])
        label_map[filename] = diagnosis

    return label_map


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_sha256_from_findings(findings: str | None) -> str | None:
    if not findings:
        return None
    match = SHA256_PATTERN.search(findings)
    return match.group(1).lower() if match else None


def iter_selected_images(images: List[Path], offset: int, limit: int) -> Iterable[Path]:
    if limit <= 0:
        return []
    return images[offset : offset + limit]


def ensure_success(response: Response, action: str) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"{action} failed with {response.status_code}: {response.text}"
        ) from exc


def check_backend(base_url: str, timeout: int) -> None:
    health_url = base_url.removesuffix("/api/v1") + "/"
    try:
        response = requests.get(health_url, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Backend is not reachable at {health_url}. Start the FastAPI server before running this script."
        ) from exc
    ensure_success(response, "Backend health check")


def load_existing_hashes(
    session: requests.Session, base_url: str, timeout: int
) -> Set[str]:
    existing_hashes: Set[str] = set()
    page = 1
    page_size = 100

    while True:
        response = session.get(
            f"{base_url}/cases/",
            params={"page": page, "page_size": page_size},
            timeout=timeout,
        )
        ensure_success(response, f"List cases page {page}")
        payload = response.json()
        cases = payload.get("cases", [])

        for case in cases:
            case_hash = extract_sha256_from_findings(case.get("findings"))
            if case_hash:
                existing_hashes.add(case_hash)

        total = int(payload.get("total", 0))
        if page * page_size >= total or not cases:
            break
        page += 1

    return existing_hashes


def create_patient(
    session: requests.Session, base_url: str, diagnosis: str, findings: str, timeout: int
) -> Dict[str, object]:
    response = session.post(
        f"{base_url}/patients/",
        json=build_patient_payload(diagnosis, findings),
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
    content_type = (
        mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    )
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
    diagnosis: str,
    findings: str,
    timeout: int,
) -> Dict[str, object]:
    response = session.put(
        f"{base_url}/cases/{case_id}",
        json={"diagnosis": diagnosis, "findings": findings},
        timeout=timeout,
    )
    ensure_success(response, f"Update case metadata for {case_id}")
    return response.json()


def embed_case(
    session: requests.Session, base_url: str, case_id: str, timeout: int
) -> Dict[str, object]:
    response = session.post(
        f"{base_url}/similarity/embed",
        params={"case_id": case_id},
        timeout=timeout,
    )
    ensure_success(response, f"Embed case {case_id}")
    return response.json()


def main() -> None:
    args = parse_args()
    images = collect_images(args.dataset_root)
    label_map = load_label_map(args.labels_file)

    if not images:
        raise RuntimeError("No gallery images found.")

    selected_images = list(iter_selected_images(images, args.offset, args.limit))
    if not selected_images:
        raise RuntimeError("No images selected. Adjust --offset or --limit.")

    print(f"Found {len(images)} images under {args.dataset_root}.")
    print(
        f"Preparing to import {len(selected_images)} record(s), starting from offset {args.offset}."
    )

    session = requests.Session()
    check_backend(args.base_url, args.timeout)
    existing_hashes = load_existing_hashes(session, args.base_url, args.timeout)
    print(f"Loaded {len(existing_hashes)} existing image hash(es) from the backend.")

    success_count = 0
    skipped_duplicates = 0
    seen_hashes = set(existing_hashes)
    for index, image_path in enumerate(selected_images, start=1):
        diagnosis = label_map.get(image_path.name)
        if not diagnosis:
            print(f"[{index}/{len(selected_images)}] Skipping unlabeled image {image_path.name}")
            continue

        image_hash = compute_file_sha256(image_path)
        if image_hash in seen_hashes:
            print(
                f"[{index}/{len(selected_images)}] Skipping duplicate {image_path.name} | SHA256: {image_hash}"
            )
            skipped_duplicates += 1
            continue

        findings = (
            f"{args.findings} Doctor diagnosis: {diagnosis}. Source file: {image_path.name} | SHA256:{image_hash}"
        )
        print(f"[{index}/{len(selected_images)}] Uploading {image_path.name} | Diagnosis: {diagnosis}")

        try:
            patient = create_patient(session, args.base_url, diagnosis, findings, args.timeout)
            case = upload_case_image(
                session, args.base_url, str(patient["id"]), image_path, args.timeout
            )
            updated_case = update_case_metadata(
                session,
                args.base_url,
                str(case["id"]),
                diagnosis,
                findings,
                args.timeout,
            )

            embed_result = None
            if args.embed:
                embed_result = embed_case(
                    session, args.base_url, str(case["id"]), args.timeout
                )

            message = (
                f"  Patient {patient['id']} | Case {updated_case['id']} | "
                f"Diagnosis: {updated_case.get('diagnosis')}"
            )
            if embed_result:
                message += f" | Vector dim: {embed_result.get('image_embedding_dim')}"
            print(message)
            seen_hashes.add(image_hash)
            success_count += 1
        except Exception as exc:
            print(f"  ERROR: {exc}")
            if not args.continue_on_error:
                raise

    print(
        f"Completed. Imported {success_count} patient/case record(s). Skipped {skipped_duplicates} duplicate image(s)."
    )


if __name__ == "__main__":
    main()
