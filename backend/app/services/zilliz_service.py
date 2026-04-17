"""
Zilliz Cloud vector database service for image-only CBIR retrieval.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import HTTPException

from app.config.settings import settings
from app.config.zilliz import get_zilliz_headers


class ZillizService:
    """Service for Zilliz Cloud operations using the REST API."""

    def __init__(self):
        self.base_url = settings.ZILLIZ_CLOUD_URI
        self.collection_name = settings.ZILLIZ_COLLECTION_NAME
        self.img_dimension = settings.ZILLIZ_IMG_DIMENSION
        self.headers = get_zilliz_headers()

    @staticmethod
    def _uuid_to_int(uuid_str: str) -> int:
        hash_val = int(hashlib.sha256(uuid_str.encode()).hexdigest(), 16)
        return hash_val % (2**63 - 1)

    def _post(
        self,
        path: str,
        payload: Dict[str, Any],
        *,
        action: str,
        timeout: int = 30,
        allow_missing: bool = False,
    ) -> Optional[Dict[str, Any]]:
        try:
            response = requests.post(
                f"{self.base_url}{path}",
                json=payload,
                headers=self.headers,
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.RequestException as exc:
            raise HTTPException(status_code=500, detail=f"{action} failed: {exc}") from exc

        if result.get("code") != 0:
            if allow_missing:
                return None
            raise HTTPException(
                status_code=500,
                detail=f"{action} failed: {result.get('message', 'Unknown error')}",
            )

        return result

    def _extract_search_results(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        data = result.get("data", [])
        if isinstance(data, list) and data:
            return data[0] if isinstance(data[0], list) else data
        return []

    def upsert_embedding(self, case_id: str, img_embedding: List[float]) -> bool:
        if len(img_embedding) != self.img_dimension:
            raise ValueError(f"Image embedding must be {self.img_dimension} dimensions")

        payload = {
            "collectionName": self.collection_name,
            "data": [
                {
                    "primary_key": self._uuid_to_int(case_id),
                    "img_emb": img_embedding,
                }
            ],
        }
        result = self._post(
            "/v2/vectordb/entities/upsert",
            payload,
            action="Upsert embedding in Zilliz",
        )
        return bool(result and result.get("code") == 0)

    def search_similar_by_image(
        self,
        img_embedding: List[float],
        *,
        top_k: int = 5,
        exclude_case_id: Optional[str] = None,
    ) -> Tuple[List[int], List[float]]:
        if len(img_embedding) != self.img_dimension:
            raise ValueError(f"Image embedding must be {self.img_dimension} dimensions")

        exclude_primary_key = self._uuid_to_int(exclude_case_id) if exclude_case_id else None
        limit = top_k + 1 if exclude_primary_key is not None else top_k

        payload = {
            "collectionName": self.collection_name,
            "data": [img_embedding],
            "annsField": "img_emb",
            "limit": limit,
            "outputFields": ["*"],
        }
        result = self._post(
            "/v2/vectordb/entities/search",
            payload,
            action="Search similar images in Zilliz",
        )

        primary_keys: List[int] = []
        scores: List[float] = []
        for item in self._extract_search_results(result or {}):
            if not isinstance(item, dict):
                continue

            primary_key = item.get("id") or item.get("primary_key")
            if primary_key is None:
                continue
            if exclude_primary_key is not None and primary_key == exclude_primary_key:
                continue

            primary_keys.append(primary_key)
            scores.append(float(item.get("distance", 0.0)))

            if len(primary_keys) >= top_k:
                break

        return primary_keys, scores

    def get_by_case_id(self, case_id: str) -> Optional[Dict[str, Any]]:
        payload = {
            "collectionName": self.collection_name,
            "id": [self._uuid_to_int(case_id)],
        }
        result = self._post(
            "/v2/vectordb/entities/get",
            payload,
            action="Get embedding from Zilliz",
            allow_missing=True,
        )
        if not result:
            return None

        data = result.get("data", [])
        return data[0] if data else None

    def delete_by_case_id(self, case_id: str) -> bool:
        primary_key = self._uuid_to_int(case_id)
        payload = {
            "collectionName": self.collection_name,
            "filter": f"primary_key in [{primary_key}]",
        }
        result = self._post(
            "/v2/vectordb/entities/delete",
            payload,
            action="Delete embedding from Zilliz",
        )
        return bool(result and result.get("code") == 0)

    def delete_batch(self, case_ids: List[str]) -> bool:
        primary_keys = ",".join(str(self._uuid_to_int(case_id)) for case_id in case_ids)
        payload = {
            "collectionName": self.collection_name,
            "filter": f"primary_key in [{primary_keys}]",
        }
        result = self._post(
            "/v2/vectordb/entities/delete",
            payload,
            action="Delete batch embeddings from Zilliz",
        )
        return bool(result and result.get("code") == 0)


zilliz_service = ZillizService()
