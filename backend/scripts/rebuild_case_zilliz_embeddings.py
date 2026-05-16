"""
Rebuild app-compatible Zilliz embeddings from Postgres cases and S3 images.

This script creates a collection whose primary key is the same stable Int64 hash
used by the backend to map search results back to Postgres case UUIDs.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import requests

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config.database import SessionLocal
from app.config.settings import settings
from app.config.zilliz import get_zilliz_headers
from app.models.models import Case
from app.services import retrieval_embedding_service, s3_service


@dataclass(frozen=True)
class CaseEmbeddingSource:
    id: str
    image_path: str


def uuid_to_int(uuid_str: str) -> int:
    hash_val = int(hashlib.sha256(uuid_str.encode()).hexdigest(), 16)
    return hash_val % (2**63 - 1)


def extract_s3_key(image_path: str) -> str:
    if image_path.startswith("http"):
        return "/".join(image_path.split("/")[3:])
    return image_path


def zilliz_endpoint(path: str) -> str:
    return f"{settings.ZILLIZ_CLOUD_URI.rstrip('/')}{path}"


def zilliz_post(path: str, payload: Dict[str, Any], *, timeout: int = 60) -> Dict[str, Any]:
    response = requests.post(
        zilliz_endpoint(path),
        json=payload,
        headers=get_zilliz_headers(),
        timeout=timeout,
    )
    response.raise_for_status()
    result = response.json()
    if result.get("code") != 0:
        raise RuntimeError(f"Zilliz request failed at {path}: {result}")
    return result


def collection_exists(collection_name: str) -> bool:
    result = zilliz_post("/v2/vectordb/collections/list", {})
    for item in result.get("data", []) or []:
        if item == collection_name:
            return True
        if isinstance(item, dict) and item.get("collectionName") == collection_name:
            return True
    return False


def drop_collection(collection_name: str) -> None:
    zilliz_post(
        "/v2/vectordb/collections/drop",
        {"collectionName": collection_name},
        timeout=120,
    )


def create_collection(collection_name: str) -> None:
    zilliz_post(
        "/v2/vectordb/collections/create",
        {
            "collectionName": collection_name,
            "description": "MedSight Postgres case image embeddings",
            "schema": {
                "fields": [
                    {
                        "fieldName": settings.ZILLIZ_PRIMARY_FIELD_NAME,
                        "dataType": "Int64",
                        "isPrimary": True,
                        "autoID": False,
                    },
                    {
                        "fieldName": settings.ZILLIZ_VECTOR_FIELD_NAME,
                        "dataType": "FloatVector",
                        "elementTypeParams": {"dim": settings.ZILLIZ_IMG_DIMENSION},
                    },
                ]
            },
            "indexParams": [
                {
                    "fieldName": settings.ZILLIZ_PRIMARY_FIELD_NAME,
                    "indexName": f"{settings.ZILLIZ_PRIMARY_FIELD_NAME}_index",
                },
                {
                    "fieldName": settings.ZILLIZ_VECTOR_FIELD_NAME,
                    "metricType": "COSINE",
                    "indexType": "AUTOINDEX",
                    "indexName": f"{settings.ZILLIZ_VECTOR_FIELD_NAME}_index",
                },
            ],
        },
        timeout=120,
    )


def ensure_collection(collection_name: str, *, recreate: bool) -> None:
    exists = collection_exists(collection_name)
    if exists and recreate:
        print(f"Dropping existing collection: {collection_name}")
        drop_collection(collection_name)
        exists = False

    if not exists:
        print(f"Creating collection: {collection_name}")
        create_collection(collection_name)


def iter_cases(limit: Optional[int]) -> Iterable[CaseEmbeddingSource]:
    db = SessionLocal()
    try:
        query = db.query(Case).order_by(Case.timestamp.asc())
        if limit:
            query = query.limit(limit)
        cases = [
            CaseEmbeddingSource(id=str(case.id), image_path=case.image_path)
            for case in query.all()
        ]
    finally:
        db.close()

    yield from cases


def clear_similarity_cache() -> None:
    db = SessionLocal()
    try:
        updated = (
            db.query(Case)
            .filter((Case.similar_cases.isnot(None)) | (Case.similarity_scores.isnot(None)))
            .update(
                {
                    Case.similar_cases: None,
                    Case.similarity_scores: None,
                },
                synchronize_session=False,
            )
        )
        db.commit()
        print(f"Cleared stale similarity cache for {updated} case(s).")
    finally:
        db.close()


def upsert_case_embedding(collection_name: str, case: CaseEmbeddingSource) -> None:
    image_bytes = s3_service.download_file(extract_s3_key(case.image_path))
    embedding = retrieval_embedding_service.generate_image_embedding(image_bytes)
    if len(embedding) != settings.ZILLIZ_IMG_DIMENSION:
        raise RuntimeError(
            f"Embedding dimension mismatch for case {case.id}: "
            f"{len(embedding)} != {settings.ZILLIZ_IMG_DIMENSION}"
        )

    zilliz_post(
        "/v2/vectordb/entities/upsert",
        {
            "collectionName": collection_name,
            "data": [
                {
                    settings.ZILLIZ_PRIMARY_FIELD_NAME: uuid_to_int(case.id),
                    settings.ZILLIZ_VECTOR_FIELD_NAME: embedding,
                }
            ],
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild Zilliz embeddings from Postgres cases.")
    parser.add_argument(
        "--collection-name",
        default=settings.ZILLIZ_COLLECTION_NAME,
        help="Target Zilliz collection name.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the target collection before inserting embeddings.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N cases.")
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Do not clear cached similar_cases/similarity_scores in Postgres.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining cases when one case fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if settings.ZILLIZ_AUTO_ID:
        raise SystemExit("Set ZILLIZ_AUTO_ID=False before rebuilding an app-compatible case collection.")

    model_info = retrieval_embedding_service.get_model_info()
    print(
        "Retrieval model loaded: "
        f"dim={model_info['embedding_dim']} input={model_info['input_size']} "
        f"providers={model_info['providers']}"
    )
    print(
        "Zilliz target: "
        f"collection={args.collection_name} primary={settings.ZILLIZ_PRIMARY_FIELD_NAME} "
        f"vector={settings.ZILLIZ_VECTOR_FIELD_NAME} dim={settings.ZILLIZ_IMG_DIMENSION}"
    )

    ensure_collection(args.collection_name, recreate=args.recreate)

    success_count = 0
    failed_count = 0
    for index, case in enumerate(iter_cases(args.limit), start=1):
        print(f"[{index}] Embedding case {case.id}")
        try:
            upsert_case_embedding(args.collection_name, case)
            success_count += 1
        except Exception as exc:
            failed_count += 1
            print(f"  ERROR: {exc}")
            if not args.continue_on_error:
                raise

    if not args.keep_cache:
        clear_similarity_cache()

    stats = zilliz_post(
        "/v2/vectordb/collections/get_stats",
        {"collectionName": args.collection_name},
    )
    print("")
    print(f"Rebuild complete. Inserted/updated {success_count} case embedding(s); failed {failed_count}.")
    print(f"Zilliz rowCount={stats.get('data', {}).get('rowCount')}")


if __name__ == "__main__":
    main()
