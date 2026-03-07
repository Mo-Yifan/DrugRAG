# src/retrieval/embedders/huggingface_embedder.py

import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from .base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)

class HuggingFaceEmbedder(BaseEmbedder):
    """
    使用 Hugging Face Sentence Transformers 的嵌入器。
    支持本地加载或自动下载模型。
    """

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5", device: str = "cpu"):
        """
        Args:
            model_name: Hugging Face 模型 ID 或本地路径
            device: 运行设备 ('cpu', 'cuda', 'mps')
        """
        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self._dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self._dimension}")

    def embed(
        self, 
        texts: List[str], 
        batch_size: int = 32, 
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        生成嵌入向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小，控制显存占用和速度
            show_progress: 是否显示进度条
        """
        if not texts:
            return []
        
        # 自动处理空文本，防止模型报错
        safe_texts = [text if text.strip() else " " for text in texts]
        
        # 调用 encode，传入 batch_size 和 progress_bar 参数
        embeddings = self.model.encode(
            safe_texts, 
            batch_size=batch_size, 
            show_progress_bar=show_progress,
            convert_to_numpy=False,  # 保持 torch tensor 或 list，稍后转
            normalize_embeddings=True  # 推荐：归一化后余弦相似度更准
        )
        
        # 转为 Python list of lists (float)
        # 如果 embeddings 是 torch tensor，先 .tolist()
        if hasattr(embeddings, 'tolist'):
            return embeddings.tolist()
        else:
            return [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embeddings]

    @property
    def dimension(self) -> int:
        return self._dimension