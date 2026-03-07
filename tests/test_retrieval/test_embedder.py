# tests/test_retrieval/test_embedder.py

import numpy as np
from src.retrieval.embedders.huggingface import HuggingFaceEmbedder


def test_hf_embedder_cpu():
    """测试嵌入模型在 CPU 上运行"""
    embedder = HuggingFaceEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )
    texts = ["Hello world", "DrugBank is useful"]
    embeddings = embedder.embed(texts)
    
    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384  # all-MiniLM-L6-v2 输出维度
    assert np.allclose(np.linalg.norm(embeddings[0]), 1.0, atol=1e-6)  # 应为单位向量