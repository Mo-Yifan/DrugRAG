# src/generation/prompts/citation_utils.py

from typing import List
from src.data import DocumentChunk

def format_citations(chunks: List[DocumentChunk]) -> str:
    """
    将检索到的 chunks 转换为带引用标记的上下文字符串。
    每个 chunk 末尾添加 <Source: DRUGBANK_ID>。
    
    Example output:
        "Aspirin is used for pain relief. <Source: DB00945>"
    """
    cited_texts = []
    for chunk in chunks:
        drugbank_id = chunk.metadata["drugbank_id"]
        # 在文本末尾插入引用
        cited_text = f"{chunk.text} <Source: {drugbank_id}>"
        cited_texts.append(cited_text)
    return "\n\n".join(cited_texts)

def extract_unique_sources(chunks: List[DocumentChunk]) -> List[str]:
    """提取所有唯一的 DrugBank ID，用于答案末尾汇总（可选）"""
    return list({chunk.metadata.drugbank_id for chunk in chunks})