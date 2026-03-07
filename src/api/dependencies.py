# src/api/dependencies.py

import os
import logging
from functools import lru_cache
from typing import Optional

from src.config import settings

# Retrieval 组件
from src.retrieval import HuggingFaceEmbedder, ChromaVectorStore

# Generation 组件
from src.generation.prompts.base_prompt import ClinicalQAPrompt
from src.generation.llm_clients import OpenAILLMClient, LocalLLMClient
from src.generation.rag_chain import RAGChain

# ✅ 新增导入：BGE Reranker
from src.retrieval.rerankers.bge_reranker import BGEReranker

logger = logging.getLogger(__name__)

@lru_cache()
def get_rag_chain() -> RAGChain:
    logger.info("🔧 初始化 RAG Chain...")

    # 1. Embedder
    embedder = HuggingFaceEmbedder(
        model_name=settings.EMBEDDING_MODEL,
        device=settings.EMBEDDING_DEVICE
    )

    # 2. Vector Store
    vectorstore = ChromaVectorStore(
        collection_name=settings.CHROMA_COLLECTION,
        persist_directory=str(settings.CHROMA_PERSIST_DIR)
    )

    # 3. LLM
    if settings.LLM_TYPE == "openai":
        llm_client = OpenAILLMClient(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY
        )
    elif settings.LLM_TYPE == "local":
        llm_client = LocalLLMClient(
            model_name=settings.LOCAL_LLM_MODEL,
            device=settings.LLM_DEVICE
        )
    else:
        raise ValueError(f"Unsupported LLM_TYPE: {settings.LLM_TYPE}")

    # 4. Prompt Builder
    prompt_builder = ClinicalQAPrompt()

    # ✅ 5. 实例化 Reranker (关键修改)
    logger.info("🚀 正在加载 BGE Reranker 模型 (这可能需要几分钟)...")
    try:
        # 使用 'BAAI/bge-reranker-v2-m3'，如果下载慢请确保设置了 HF_ENDPOINT 环境变量
        reranker = BGEReranker(model_name='BAAI/bge-reranker-v2-m3')
        logger.info("✅ BGE Reranker 加载成功!")
    except Exception as e:
        logger.error(f"❌ 加载 BGE Reranker 失败: {e}")
        logger.warning("⚠️ 将降级使用 None (不启用重排序)")
        reranker = None

    # 6. Assemble Chain
    rag_chain = RAGChain(
        embedder=embedder,
        vectorstore=vectorstore,
        llm=llm_client,
        prompt_builder=prompt_builder,
        reranker=reranker,  # <--- 传入刚才实例化的 reranker
        top_k=settings.FINAL_ANSWER_TOP_K
    )

    logger.info("✅ RAG Chain 初始化完成!")
    return rag_chain