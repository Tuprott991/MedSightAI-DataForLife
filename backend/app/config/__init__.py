"""
Configuration module.

Imports that touch external services are kept lazy so utility scripts can load
settings without requiring every runtime dependency.
"""
from app.config.settings import settings
from app.config.zilliz import get_zilliz_config, get_zilliz_headers


def get_db(*args, **kwargs):
    from app.config.database import get_db as _get_db

    return _get_db(*args, **kwargs)


def init_db(*args, **kwargs):
    from app.config.database import init_db as _init_db

    return _init_db(*args, **kwargs)


def get_s3_client(*args, **kwargs):
    from app.config.s3 import get_s3_client as _get_s3_client

    return _get_s3_client(*args, **kwargs)


def get_s3_resource(*args, **kwargs):
    from app.config.s3 import get_s3_resource as _get_s3_resource

    return _get_s3_resource(*args, **kwargs)


__all__ = [
    "settings",
    "get_db",
    "init_db",
    "get_s3_client",
    "get_s3_resource",
    "get_zilliz_headers",
    "get_zilliz_config",
]
