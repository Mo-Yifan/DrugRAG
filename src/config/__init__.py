# src/config/__init__.py

"""
配置模块统一入口。
"""

from .settings import settings
from .paths import (
    ROOT_DIR,
    DATA_DIR,
    DRUGBANK_XML_PATH,
    DRUGBANK_JSON_PATH,
    VECTOR_STORE_DIR,
    CHROMA_PERSIST_DIR,
    CACHE_DIR,
    LOG_DIR
)

__all__ = [
    "settings",
    "ROOT_DIR",
    "DATA_DIR",
    "DRUGBANK_XML_PATH",
    "DRUGBANK_JSON_PATH"
    "VECTOR_STORE_DIR",
    "CHROMA_PERSIST_DIR",
    "CACHE_DIR",
    "LOG_DIR"
]