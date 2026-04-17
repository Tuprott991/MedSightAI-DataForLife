"""
Bootstrap PostgreSQL and Zilliz Cloud for the target MedSight schema.

This script is intentionally standalone so it does not depend on the backend's
full application settings. That keeps setup simple when you only want to
initialize Neon and Zilliz first.

Usage:
    python scripts/bootstrap_datastores.py
    python scripts/bootstrap_datastores.py --postgres-only
    python scripts/bootstrap_datastores.py --zilliz-only
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable

import requests
from sqlalchemy import create_engine, text


ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"


def load_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        values[key] = value

    return values


ENV_FILE_VALUES = load_env_file(ENV_PATH)


def env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name) or ENV_FILE_VALUES.get(name) or default


def require_vars(names: Iterable[str]) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    missing = []

    for name in names:
        value = env(name)
        if value is None or value == "":
            missing.append(name)
        else:
            resolved[name] = value

    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            f"Missing required environment variables: {joined}. "
            f"Set them in {ENV_PATH} or your shell environment."
        )

    return resolved


POSTGRES_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'report_status_enum') THEN
        CREATE TYPE report_status_enum AS ENUM ('draft', 'reviewed', 'final');
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'chat_session_type_enum') THEN
        CREATE TYPE chat_session_type_enum AS ENUM ('practice', 'tutoring');
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'chat_sender_enum') THEN
        CREATE TYPE chat_sender_enum AS ENUM ('user', 'ai');
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS patient (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    age INTEGER,
    gender TEXT,
    history JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES patient(id) ON DELETE CASCADE,
    image_path TEXT NOT NULL,
    processed_img_path TEXT,
    similar_cases JSON,
    similarity_scores JSON,
    "timestamp" TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ai_result (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    predictions JSONB,
    bounding_boxes JSONB,
    gradcam_path TEXT,
    confidence_scores JSONB,
    concepts JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS report (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    model_report TEXT,
    doctor_report TEXT,
    status report_status_enum NOT NULL DEFAULT 'draft',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_session (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    case_id UUID REFERENCES cases(id) ON DELETE SET NULL,
    session_type chat_session_type_enum NOT NULL,
    score DOUBLE PRECISION,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS chat_message (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES chat_session(id) ON DELETE CASCADE,
    sender chat_sender_enum NOT NULL,
    message TEXT NOT NULL,
    "timestamp" TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cases_patient_id ON cases(patient_id);
CREATE INDEX IF NOT EXISTS idx_cases_timestamp ON cases("timestamp");
CREATE INDEX IF NOT EXISTS idx_ai_result_case_id ON ai_result(case_id);
CREATE INDEX IF NOT EXISTS idx_report_case_id ON report(case_id);
CREATE INDEX IF NOT EXISTS idx_chat_session_case_id ON chat_session(case_id);
CREATE INDEX IF NOT EXISTS idx_chat_session_user_id ON chat_session(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_message_session_id ON chat_message(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_message_timestamp ON chat_message("timestamp");

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS report_set_updated_at ON report;
CREATE TRIGGER report_set_updated_at
BEFORE UPDATE ON report
FOR EACH ROW
EXECUTE FUNCTION set_updated_at();
"""


def bootstrap_postgres() -> None:
    config = require_vars(["DATABASE_URL"])
    engine = create_engine(config["DATABASE_URL"], pool_pre_ping=True)

    print("Connecting to PostgreSQL...")
    with engine.begin() as connection:
        print("Creating PostgreSQL enums, tables, indexes, and triggers...")
        connection.execute(text(POSTGRES_SCHEMA_SQL))

        tables = [
            "patient",
            "cases",
            "ai_result",
            "report",
            "chat_session",
            "chat_message",
        ]
        for table in tables:
            exists = connection.execute(
                text("SELECT to_regclass(:table_name)"),
                {"table_name": table},
            ).scalar_one()
            print(f"  - {table}: {'ok' if exists else 'missing'}")


def zilliz_headers(api_key: str) -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def zilliz_endpoint(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def zilliz_collection_exists(base_url: str, api_key: str, collection_name: str) -> bool:
    response = requests.post(
        zilliz_endpoint(base_url, "/v2/vectordb/collections/list"),
        json={},
        headers=zilliz_headers(api_key),
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()
    if payload.get("code") != 0:
        raise RuntimeError(f"Failed to list collections: {payload}")

    items = payload.get("data", []) or []
    for item in items:
        if isinstance(item, str):
            if item == collection_name:
                return True
            continue

        if isinstance(item, dict):
            if item.get("collectionName") == collection_name or item.get("name") == collection_name:
                return True
    return False


def bootstrap_zilliz() -> None:
    config = require_vars(["ZILLIZ_CLOUD_URI"])
    api_key = env("ZILLIZ_CLOUD_API_KEY", "") or ""
    collection_name = env("ZILLIZ_COLLECTION_NAME", "med_vector")
    img_dim = int(env("ZILLIZ_IMG_DIMENSION", "1024"))

    print(f"Checking Zilliz collection '{collection_name}'...")
    if zilliz_collection_exists(config["ZILLIZ_CLOUD_URI"], api_key, collection_name):
        print("  - collection already exists")
        print("  - existing collections are not mutated by this script; drop or rename the collection to apply schema changes")
        return

    payload = {
        "collectionName": collection_name,
        "description": "MedSight vectors for similar case retrieval",
        "schema": {
            "fields": [
                {
                    "fieldName": "primary_key",
                    "dataType": "Int64",
                    "isPrimary": True,
                    "autoID": False,
                },
                {
                    "fieldName": "img_emb",
                    "dataType": "FloatVector",
                    "elementTypeParams": {
                        "dim": img_dim,
                    },
                },
            ]
        },
        "indexParams": [
            {
                "fieldName": "primary_key",
                "indexName": "primary_key_index",
            },
            {
                "fieldName": "img_emb",
                "metricType": "COSINE",
                "indexType": "AUTOINDEX",
                "indexName": "img_emb_index",
            },
        ],
    }

    print("Creating Zilliz collection...")
    response = requests.post(
        zilliz_endpoint(config["ZILLIZ_CLOUD_URI"], "/v2/vectordb/collections/create"),
        json=payload,
        headers=zilliz_headers(api_key),
        timeout=60,
    )
    response.raise_for_status()

    result = response.json()
    if result.get("code") != 0:
        raise RuntimeError(f"Failed to create collection: {result}")

    print("  - collection created")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap PostgreSQL and Zilliz for MedSight.")
    parser.add_argument("--postgres-only", action="store_true", help="Only create PostgreSQL schema.")
    parser.add_argument("--zilliz-only", action="store_true", help="Only create Zilliz collection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.postgres_only and args.zilliz_only:
        raise SystemExit("Choose either --postgres-only or --zilliz-only, not both.")

    run_postgres = not args.zilliz_only
    run_zilliz = not args.postgres_only

    if run_postgres:
        bootstrap_postgres()

    if run_zilliz:
        bootstrap_zilliz()

    print("")
    print("Bootstrap complete.")
    print("Note: Zilliz primary_key is Int64, so your application should map case UUIDs to a stable Int64 value.")


if __name__ == "__main__":
    main()
