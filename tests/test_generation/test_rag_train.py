# tests/test_generation/test_rag_chain.py

from unittest.mock import MagicMock
from src.generation.rag_chain import RAGChain


def test_rag_chain_e2e(mock_openai_client, sample_question):
    """端到端测试 RAG Chain（使用 mock）"""
    # Mock 所有依赖
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.5, 0.5]]
    
    mock_vs = MagicMock()
    mock_vs.search.return_value = [
        {"text": "Half-life is 15 min.", "metadata": {"drugbank_id": "DB00945"}}
    ]
    
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        {"text": "Half-life is 15 min.", "metadata": {"drugbank_id": "DB00945"}}
    ]
    
    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build.return_value = "Mocked prompt"
    
    # 组装 RAG Chain
    chain = RAGChain(
        embedder=mock_embedder,
        vectorstore=mock_vs,
        reranker=mock_reranker,
        llm=mock_openai_client,
        prompt_builder=mock_prompt_builder
    )
    
    # 执行
    result = chain.invoke(sample_question)
    
    # 验证
    assert "15" in result["answer"]
    assert "<Source: DB00945>" in result["answer"]
    mock_openai_client.generate.assert_called_once()