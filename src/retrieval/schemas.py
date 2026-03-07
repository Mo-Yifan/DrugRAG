# src/retrieval/schemas.py
"""
统一数据模型（Zero-dependency）
==============================
所有模块共用的核心数据结构，不 import 任何 retrieval 内部模块。
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ================
# 🧬 原始药物数据
# ================

class Drug(BaseModel):
    drugbank_id: str
    name: str
    description: Optional[str] = None
    indications: Optional[str] = None
    pharmacodynamics: Optional[str] = None
    mechanism_of_action: Optional[str] = None
    toxicity: Optional[str] = None
    metabolism: Optional[str] = None
    half_life: Optional[str] = None
    fda_approved: bool = False
    synonyms: List[str] = Field(default_factory=list)

    def has_clinical_text(self) -> bool:
        fields = [
            self.description, self.indications, self.pharmacodynamics,
            self.mechanism_of_action, self.toxicity, self.metabolism, self.half_life
        ]
        return any(field and field.strip() for field in fields)


# ================
# 📄 PubMed 文章
# ================

class PubmedArticle(BaseModel):  # 👈 关键修复：继承 BaseModel
    pmid: str
    title: str
    abstract: str
    drug_mentions: List[str] = Field(default_factory=list)  # 👈 安全默认值


# ================
# ✂️ 文本块（Chunk）
# ================

class TextChunk(BaseModel):
    """切分后的文本单元"""
    text: str
    metadata: Dict[str, Any]  # 包含 drugbank_id, field, drug_name 等
    source_id: Optional[str] = None  # 如 "DB00001_description"
    embedding: Optional[List[float]] = None


# ================
# 🧠 嵌入结果
# ================

class EmbeddingResult(BaseModel):
    texts: List[str]
    embeddings: List[List[float]]
    metadatas: List[Dict[str, Any]]


# ================
# 🔍 检索结果
# ================

class RetrievedChunk(BaseModel):
    text: str
    metadata: Dict[str, Any]
    score: float  # 相似度分数