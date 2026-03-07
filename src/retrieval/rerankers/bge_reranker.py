# src/retrieval/rerankers/bge_reranker.py

import logging
import json
import re
from typing import List, Set
from .base_reranker import BaseReranker
from src.retrieval.schemas import RetrievedChunk
from FlagEmbedding import FlagReranker

logger = logging.getLogger(__name__)


class BGEReranker(BaseReranker):
    """
    基于 BGE Reranker 的语义重排序器。
    支持关键词匹配增强（如 drug_name / synonyms 匹配）。
    """

    def __init__(self, model_name: str = 'BAAI/bge-reranker-v2-m3'):
        logger.info(f"Loading BGE Reranker: {model_name}")
        self.reranker = FlagReranker(model_name, use_fp16=True)
        logger.info("BGE Reranker loaded successfully.")

    def _extract_query_keywords(self, query: str) -> Set[str]:
        """ 提取问题中的关键词（过滤标点，长度 > 2） """
        cleaned = re.sub(r'[^\w\s]', ' ', query)
        return {word.lower() for word in cleaned.split() if len(word) > 2}

    def _calculate_priority_boost(self, query_keywords: Set[str], chunk: RetrievedChunk) -> float:
        """ 若 chunk 的 drug_name 或 synonyms 匹配 query 关键词，给予分数奖励 """
        if not query_keywords:
            return 0.0

        drug_name = chunk.metadata.get('drug_name', '').lower()
        synonyms_str = chunk.metadata.get('synonyms', '[]')
        
        try:
            synonyms = json.loads(synonyms_str)
        except (json.JSONDecodeError, TypeError):
            synonyms = []
        
        if not isinstance(synonyms, list):
            synonyms = []
        
        synonym_set = {s.lower() for s in synonyms if isinstance(s, str)}

        for keyword in query_keywords:
            if keyword in synonym_set:
                return 1.0
            if keyword == drug_name:
                return 0.8
        return 0.0

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = 5
    ) -> List[RetrievedChunk]:
        if not chunks:
            return []

        # 1. 提取关键词
        query_keywords = self._extract_query_keywords(query)

        # 2. BGE 原始打分
        pairs = [[query, chunk.text] for chunk in chunks]
        scores = self.reranker.compute_score(pairs, normalize=True)
        
        if isinstance(scores, tuple):
            scores = scores[0]
        
        min_len = min(len(scores), len(chunks))
        scores = scores[:min_len]
        chunks = chunks[:min_len]

        # 3. 添加优先级奖励并赋值
        for i, chunk in enumerate(chunks):
            original_score = float(scores[i])
            boost = self._calculate_priority_boost(query_keywords, chunk)
            chunk.score = original_score + boost

        # 4. 按 score 降序排序
        chunks.sort(key=lambda x: x.score, reverse=True)

        # 5. （可选）打印排行榜（生产环境可关闭）
        if logger.isEnabledFor(logging.INFO):
            logger.info("=" * 60)
            logger.info(f"🏆 Rerank 排行榜 (Query: {query[:40]}...)")
            logger.info("-" * 60)
            for i, chunk in enumerate(chunks[:5]):  # 只打前5
                drug_name = chunk.metadata.get('drug_name', 'Unknown')
                field = chunk.metadata.get('field', 'text')
                marker = "⚡" if chunk.score > 1.0 else " "
                logger.info(f"{marker} #{i+1}: [{chunk.score:.4f}] {drug_name} ({field})")
            logger.info("=" * 60)
            if chunks:
                logger.info(f"✅ Top result: {chunks[0].metadata.get('drug_name', 'Unknown')} (Score: {chunks[0].score:.4f})")

        return chunks[:top_k]