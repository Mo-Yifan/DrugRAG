# src/retrieval/vectorstores/chroma_store.py
import logging
import chromadb
from typing import List, Optional
from .base_vectorstore import BaseVectorStore
from ..schemas import TextChunk, RetrievedChunk  # ✅ 使用 retrieval.schemas

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """
    使用 ChromaDB 作为向量存储后端。
    优势：原生元数据过滤、自动 ID、持久化简单。
    """

    def __init__(
        self,
        collection_name: str = "drugbank_rag",
        persist_directory: str = "./artifacts/vectors/chroma"
    ):
        self.client = chromadb.PersistentClient(path=persist_directory)
        # 注意：我们自己提供 embedding，所以禁用内部 embedding function
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None
        )
        logger.info(f"ChromaDB collection '{collection_name}' ready.")

    def add_chunks(self, chunks: List[TextChunk]) -> None:
        """
        实现基类抽象方法：将 TextChunk 列表存入 Chroma
        【新增】支持自动分批插入，避免超过最大 Batch Size 限制
        """
        if not chunks:
            return

        # 验证所有 chunk 都有 embedding
        for i, chunk in enumerate(chunks):
            if chunk.embedding is None:
                raise ValueError(
                    f"TextChunk[{i}] missing embedding. "
                    "Did you forget to call embedder and assign chunk.embedding?"
                )

        # === 关键修改：设置批次大小 ===
        # ChromaDB 默认限制约为 5000-6000，为了安全设为 1000
        BATCH_SIZE = 1000
        
        total_count = len(chunks)
        logger.info(f"💾 开始存入 {total_count} 个文档 (分批处理，每批 {BATCH_SIZE} 条)...")

        for i in range(0, total_count, BATCH_SIZE):
            batch_chunks = chunks[i : i + BATCH_SIZE]
            
            # 生成唯一 IDs
            ids = [
                f"{chunk.metadata.get('drugbank_id', 'unknown')}_{chunk.metadata.get('field', 'text')}_{idx}"
                for idx, chunk in enumerate(batch_chunks)
                # 注意：这里的 idx 是批次内的索引，可能导致全局 ID 重复！
                # ✅ 修正：使用全局索引 i + idx
            ]
            
            # ✅ 修正 ID 生成逻辑（使用全局索引避免重复）
            ids = []
            texts = []
            metadatas = []
            embeddings = []
            
            for idx, chunk in enumerate(batch_chunks):
                global_idx = i + idx
                ids.append(f"{chunk.metadata.get('drugbank_id', 'unk')}_{chunk.metadata.get('field', 'txt')}_{global_idx}")
                texts.append(chunk.text)
                metadatas.append(chunk.metadata)
                embeddings.append(chunk.embedding)

            try:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas
                )
                logger.info(f"   ✅ 已存入批次 {i//BATCH_SIZE + 1}: {len(batch_chunks)} 条 (累计 {min(i+BATCH_SIZE, total_count)}/{total_count})")
            except Exception as e:
                logger.error(f"❌ 批次 {i//BATCH_SIZE + 1} 插入失败: {e}")
                raise e

        logger.info(f"🎉 所有 {total_count} 个文档已成功存入 ChromaDB!")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_criteria: Optional[dict] = None
    ) -> List[RetrievedChunk]:
        """
        实现基类 search 方法（参数名已对齐）
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_criteria  # Chroma 原生支持
        )

        retrieved_chunks = []
        for i in range(len(results["ids"][0])):
            text = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]  # Chroma 返回的是距离（越小越相似）

            # 转换为相似度分数 [0,1]（可选，也可直接返回 distance）
            score = 1.0 / (1.0 + distance)

            retrieved_chunks.append(RetrievedChunk(
                text=text,
                metadata=metadata,
                score=score
            ))

        return retrieved_chunks

    def save(self) -> None:
        # PersistentClient 自动持久化
        logger.info("ChromaDB is automatically persisted.")

    def load(self) -> None:
        pass