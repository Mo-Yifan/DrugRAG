# tests/conftest.py

import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock

# 将项目根目录加入 Python 路径
ROOT_DIR = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT_DIR))

# 设置测试环境变量
os.environ.update({
    "LLM_TYPE": "openai",
    "OPENAI_API_KEY": "test-key",
    "EMBEDDING_DEVICE": "cpu",
    "RERANKER_DEVICE": "cpu",
    "LLM_DEVICE": "cpu",
    "DEBUG": "true"
})

@pytest.fixture
def mock_openai_client():
    """模拟 OpenAI 客户端"""
    mock = MagicMock()
    mock.generate.return_value = "阿司匹林的半衰期约为 15-20 分钟。<Source: DB00945>"
    return mock

@pytest.fixture
def sample_drug_data():
    """返回一个简化 Drug 对象（字典形式）用于测试"""
    return {
        "drugbank_id": "DB00945",
        "name": "Aspirin",
        "description": "Aspirin is a nonsteroidal anti-inflammatory drug.",
        "indications": "Used for pain, fever, and inflammation.",
        "fda_approved": True
    }

@pytest.fixture
def sample_question():
    return "阿司匹林的半衰期是多少？"