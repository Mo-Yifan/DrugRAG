#!/usr/bin/env python3
# scripts/ingest.py
"""DrugBank 数据注入脚本
======================
- 使用 src.data.loaders 加载 XML/JSON
- 使用 src.data.chunkers 切分文本
- 所有数据结构来自 src.retrieval.schemas
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到 Python 路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# 🔑 关键：所有数据结构来自 retrieval.schemas
from src.retrieval.schemas import Drug, TextChunk  # TextChunk 现在包含 .embedding 字段

# data 模块（loader + chunker）
from src.data.loaders import DrugBankJSONLoader
from src.data.chunkers import SemanticChunker, MedicalSectionChunker

# retrieval 模块（嵌入 + 向量库）
from src.retrieval import HuggingFaceEmbedder, ChromaVectorStore

# 配置
from src.config import (
    settings,
    DRUGBANK_XML_PATH,
    DRUGBANK_JSON_PATH,
    CHROMA_PERSIST_DIR
)

# 日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def select_input_file() -> Path:
    """优先选择 JSON，其次 XML"""
    if DRUGBANK_JSON_PATH.exists():
        return DRUGBANK_JSON_PATH
    elif DRUGBANK_XML_PATH.exists():
        return DRUGBANK_XML_PATH
    else:
        raise FileNotFoundError(
            "❌ 未找到 DrugBank 数据文件。\n"
            f"请提供以下任一文件:\n"
            f"  - {DRUGBANK_XML_PATH}\n"
            f"  - {DRUGBANK_JSON_PATH}"
        )


def get_loader(file_path: Path):
    suffix = file_path.suffix.lower()
    if suffix == ".xml":
        pass  # TODO: 实现 XML loader
    elif suffix == ".json":
        return DrugBankJSONLoader()
    else:
        raise ValueError(f"不支持的格式: {suffix}")


def main():
    logger.info("🚀 开始 DrugBank 数据注入流程...")

    # === 1. 选择输入文件 ===
    try:
        input_path = select_input_file()
        logger.info(f"📄 选中文件: {input_path}")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # === 2. 初始化组件 ===
    loader = get_loader(input_path)
    
    chunker_type = getattr(settings, "CHUNKER_TYPE", "semantic")
    if chunker_type == "medical_section":
        chunker = MedicalSectionChunker(max_length=512, overlap=50)
    else:
        chunker = SemanticChunker(max_chunk_size=512, overlap=50)

    embedder = HuggingFaceEmbedder(
        model_name=settings.EMBEDDING_MODEL,
        device=settings.EMBEDDING_DEVICE
    )
    vectorstore = ChromaVectorStore(
        collection_name=settings.CHROMA_COLLECTION,
        persist_directory=CHROMA_PERSIST_DIR
    )

    # === 3. 加载与切分 ===
    logger.info("🔍 加载数据...")
    drugs: list[Drug] = loader.load(input_path)  # 返回 src.retrieval.schemas.Drug
    logger.info(f"✅ 成功加载 {len(drugs)} 种药物")

    logger.info("✂️ 切分文本...")
    all_chunks: list[TextChunk] = chunker.split(drugs)  # 返回 src.retrieval.schemas.TextChunk
    logger.info(f"✅ 生成 {len(all_chunks)} 个文本块")

    # === 4. 生成嵌入并向 TextChunk 注入 embedding ===
    texts = [chunk.text for chunk in all_chunks]
    logger.info("🧠 生成嵌入...")
    embeddings = embedder.embed(texts, batch_size=settings.EMBEDDING_BATCH_SIZE)

    # 🔑 方案 A 第二步：将 embedding 注入每个 TextChunk
    for chunk, emb in zip(all_chunks, embeddings):
        chunk.embedding = emb  # ← 关键修改！

    # === 5. 存入向量库（使用统一的 add_chunks 接口）===
    logger.info("💾 存入向量库...")
    vectorstore.add_chunks(all_chunks)  # ← 现在可以安全调用！

    logger.info("🎉 注入完成！")
    logger.info(f" 药物数: {len(drugs)} | 文本块: {len(all_chunks)}")
    logger.info(f" 向量库存储于: {CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    main()