# tests/test_retrieval/test_vectorstore.py

import tempfile
from src.retrieval.vectorstores.chroma import ChromaVectorStore


def test_chroma_add_and_search():
    """测试 Chroma 向量库的添加与检索"""
    with tempfile.TemporaryDirectory() as tmpdir:
        vs = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=tmpdir
        )
        
        texts = ["Aspirin treats pain.", "Insulin manages diabetes."]
        embeddings = [[0.1, 0.9], [0.8, 0.2]]  # 模拟嵌入
        metadatas = [{"drug": "Aspirin"}, {"drug": "Insulin"}]
        
        vs.add_texts(texts, embeddings, metadatas)
        
        # 查询最相似
        results = vs.search(query_embedding=[0.11, 0.89], k=1)
        assert len(results) == 1
        assert "Aspirin" in results[0]["text"]
        assert results[0]["metadata"]["drug"] == "Aspirin"