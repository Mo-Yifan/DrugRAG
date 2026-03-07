# tests/test_config/test_settings.py

from src.config import settings
import pytest


def test_settings_loaded():
    """测试配置是否正确加载"""
    assert settings.PROJECT_NAME == "DrugBank Clinical RAG"
    assert settings.LLM_TYPE in ["openai", "local"]
    assert settings.EMBEDDING_DEVICE == "cpu"  # 来自 conftest 的环境变量


def test_openai_key_required():
    """测试当 LLM_TYPE=openai 时，API Key 必须存在"""
    assert settings.OPENAI_API_KEY == "test-key"

    # 模拟无 key 的情况
    from pydantic import ValidationError
    import os
    old_key = os.environ.pop("OPENAI_API_KEY", None)
    try:
        from src.config.settings import Settings
        with pytest.raises(ValidationError):
            Settings(LLM_TYPE="openai")
    finally:
        if old_key:
            os.environ["OPENAI_API_KEY"] = old_key