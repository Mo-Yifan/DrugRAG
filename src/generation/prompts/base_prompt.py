# src/generation/prompts/base_prompt.py

import os
from pathlib import Path
from typing import List
from .citation_utils import format_citations
from src.data import DocumentChunk

class ClinicalQAPrompt:
    """
    加载并格式化临床问答 Prompt。
    """

    def __init__(self, template_path: str = None):
        if template_path is None:
            template_path = Path(__file__).parent / "clinical_qa_prompt.txt"
        with open(template_path, "r", encoding="utf-8") as f:
            self.template = f.read().strip()

    def build(self, question: str, retrieved_chunks: List[DocumentChunk]) -> str:
        """
        构建完整 prompt。
        
        Args:
            question: 用户问题
            retrieved_chunks: 检索到的 DocumentChunk 列表
            
        Returns:
            格式化后的 prompt 字符串
        """
        context = format_citations(retrieved_chunks)
        return self.template.format(context=context, question=question)