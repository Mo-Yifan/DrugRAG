# src/generation/rag_chain.py

import logging
import re
from typing import List, Optional, Dict, Any, Set
from src.retrieval.schemas import TextChunk, RetrievedChunk
from src.retrieval.embedders.base_embedder import BaseEmbedder
from src.retrieval.vectorstores.base_vectorstore import BaseVectorStore
# 1. 导入 Reranker 基类 (用于类型提示，可选)
# 如果你的 dependencies.py 直接传入了具体实例，这里其实不需要导入具体类，
# 但为了代码清晰，我们可以导入基类作为类型提示。
from src.retrieval.rerankers.base_reranker import BaseReranker 
from src.generation.prompts.base_prompt import ClinicalQAPrompt
from src.generation.llm_clients.base_client import BaseLLMClient

logger = logging.getLogger(__name__)


class RAGChain:
    """ RAG 核心链路： Embedding -> Retrieval -> [Rerank] -> Prompt Building -> LLM Generation """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vectorstore: BaseVectorStore,
        llm: Any,
        prompt_builder: ClinicalQAPrompt,
        reranker: Optional[BaseReranker] = None,  # 接收 Reranker 实例 (可以是 BGEReranker 或 NoneReranker)
        top_k: int = 3
    ):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.reranker = reranker
        self.top_k = top_k

        # 2. 设置检索倍数
        # 如果启用了 Rerank，我们需要先多取一些候选集 (例如 top_k=3, 则取 3*5=15 个)
        # 让 Rerank 模型从这 9 个里挑出最好的 3 个。
        # 如果没有 Rerank (NoneReranker 或 None)，则直接取 top_k。
        if self.reranker and not isinstance(self.reranker, type(None)):
            # 简单判断：如果传入的不是 None 对象，且不是空的占位符
            # 注意：如果你的 NoneReranker 是一个类实例，需要确保它有个标识或者我们就默认只要不是 None 就开启
            # 更稳妥的方式：检查类名或特定属性，或者直接默认 multiplier=3 (如果 reranker 存在)
            self.retrieve_multiplier = 5
            logger.info(f"🚀 Reranker 已启用: {type(self.reranker).__name__}")
        else:
            self.retrieve_multiplier = 1
            logger.info("⚠️ Reranker 未启用 (或为 NoneReranker)，将直接使用向量检索结果。")

        logger.info(f"RAGChain initialized with top_k={self.top_k}, retrieve_multiplier={self.retrieve_multiplier}")

    def _extract_query_keywords(self, query: str) -> Set[str]:
        """ 提取问题中的关键词（模仿 BGEReranker 的逻辑）"""
        cleaned = re.sub(r'[^\w\s]', ' ', query)
        keywords = {word.lower() for word in cleaned.split() if len(word) > 2}
        return keywords

    def invoke(self, question: str, fda_only: bool = False) -> Dict[str, Any]:
        logger.info(f"🔍 处理问题: {question[:50]}...")
        try:
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # 🔥 新增多路检索逻辑（启动多路检测）
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            use_multi_hop = True  # 控制开关

            if use_multi_hop:
                query_keywords = self._extract_query_keywords(question)
                logger.debug(f"🔍 多路检测：提取到关键词 {query_keywords}")

                # 只有 ≥2 个关键词才触发多路
                if len(query_keywords) >= 2:
                    logger.info(f"🔄 启动多路检索，关键词: {list(query_keywords)[:5]}")
                    all_retrieved_chunks: List[RetrievedChunk] = []
                    seen_texts: Set[str] = set()

                    # 为每个关键词生成子查询
                    for kw in query_keywords:
                        sub_query = f"interactions of {kw}"
                        logger.debug(f"  → 子查询: {sub_query}")
                        sub_embedding = self.embedder.embed([sub_query])[0]
                        search_top_k = self.top_k * self.retrieve_multiplier
                        filter_criteria = {"fda_approved": True} if fda_only else None
                        chunks = self.vectorstore.search(
                            query_embedding=sub_embedding,
                            top_k=search_top_k,
                            filter_criteria=filter_criteria
                        )
                        # 去重合并
                        for chunk in chunks:
                            if chunk.text not in seen_texts:
                                seen_texts.add(chunk.text)
                                all_retrieved_chunks.append(chunk)

                    retrieved_chunks = all_retrieved_chunks
                    logger.info(f"✅ 多路检索完成，共召回 {len(retrieved_chunks)} 个唯一片段")
                else:
                    # 关键词不足，回退到单路
                    query_embedding = self.embedder.embed([question])[0]
                    search_top_k = self.top_k * self.retrieve_multiplier
                    filter_criteria = {"fda_approved": True} if fda_only else None
                    retrieved_chunks = self.vectorstore.search(
                        query_embedding=query_embedding,
                        top_k=search_top_k,
                        filter_criteria=filter_criteria
                    )
            else:
                # 原始单路逻辑
                query_embedding = self.embedder.embed([question])[0]
                search_top_k = self.top_k * self.retrieve_multiplier
                filter_criteria = {"fda_approved": True} if fda_only else None
                retrieved_chunks = self.vectorstore.search(
                    query_embedding=query_embedding,
                    top_k=search_top_k,
                    filter_criteria=filter_criteria
                )

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # 🔥 多路逻辑结束，后续流程完全不变
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            logger.info(f"检索到 {len(retrieved_chunks)} 个原始结果")
            if not retrieved_chunks:
                return {
                    "answer": "未在数据库中找到相关信息。",
                    "citations": [],
                    "question": question
                }

            # 3. 重排序 (Rerank) - 🔥 核心修复点
            if self.reranker:
                logger.info(f"🔄 启动重排序 ({type(self.reranker).__name__}): {len(retrieved_chunks)} -> {self.top_k} ...")
                try:
                    # 调用你的 reranker.rerank 方法
                    # 假设你的接口是：rerank(query: str, chunks: List[RetrievedChunk], top_k: int)
                    retrieved_chunks = self.reranker.rerank(
                        query=question,
                        chunks=retrieved_chunks,
                        top_k=self.top_k
                    )
                    if retrieved_chunks:
                        logger.info(f"✅ 重排序完成。新 Top 1: {retrieved_chunks[0].metadata.get('drug_name')} (Score: {retrieved_chunks[0].score:.4f})")
                    else:
                        logger.warning("⚠️ 重排序后结果为空！")
                except Exception as e:
                    logger.error(f"❌ Rerank 过程出错: {e}", exc_info=True)
                    logger.warning(f"⚠️ Rerank 失败，降级使用原始检索结果的前 {self.top_k} 个。")
                    # 降级：直接截取前 top_k 个
                    retrieved_chunks = retrieved_chunks[:self.top_k]
            else:
                # 如果没有 reranker，直接截断
                retrieved_chunks = retrieved_chunks[:self.top_k]
                logger.info(f"ℹ️ 未启用 Rerank，直接截取前 {self.top_k} 个结果。")

            # 在构建 prompt 前
            final_context_text = "\n".join([f"[{i}] {chunk.text}" for i, chunk in enumerate(retrieved_chunks)])
            logger.warning(f"🎯 FINAL CONTEXT SENT TO LLM:\n{final_context_text}")

            # 4. 构建 Prompt
            # 将 RetrievedChunk 转换为 TextChunk (如果 prompt_builder 需要)
            context_chunks = [
                TextChunk(text=c.text, metadata=c.metadata) for c in retrieved_chunks
            ]
            prompt = self.prompt_builder.build(question, context_chunks)
            logger.debug(f"Prompt 构建完成，长度: {len(prompt)} chars")

            # 5. 调用 LLM 生成回答
            logger.debug("调用 LLM 生成回答...")
            answer = self.llm.generate(prompt)

            # 6. 提取引用信息 (用于前端展示)
            citations = [
                {
                    "drugbank_id": c.metadata.get("drugbank_id", "Unknown"),
                    "drug_name": c.metadata.get("drug_name", "Unknown"),
                    "field": c.metadata.get("field", "Unknown"),
                    "score": float(getattr(c, 'score', 0.0))
                }
                for c in retrieved_chunks
            ]

            return {
                "answer": answer,
                "citations": citations,
                "question": question
            }

        except Exception as e:
            logger.error(f"RAGChain invoke 发生严重错误: {e}", exc_info=True)
            raise e