# tests/test_api/test_routes.py

import pytest
from fastapi.testclient import TestClient
from src.api.app import app
import MagicMock


@pytest.fixture
def client():
    return TestClient(app)


def test_web_ui(client):
    """测试 Web 界面可访问"""
    response = client.get("/")
    assert response.status_code == 200
    assert "DrugBank 临床问答助手" in response.text


def test_api_query_success(client, monkeypatch):
    """测试 API 查询成功（需 mock RAGChain）"""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = {
        "answer": "半衰期为 15 分钟。<Source: DB00945>"
    }
    
    # 替换依赖
    from src.api.dependencies import get_rag_chain
    monkeypatch.setattr("src.api.dependencies.get_rag_chain", lambda: mock_chain)
    
    response = client.post(
        "/api/query",
        json={"question": "阿司匹林半衰期？"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "半衰期" in data["answer"]
    assert "<Source: DB00945>" in data["answer"]