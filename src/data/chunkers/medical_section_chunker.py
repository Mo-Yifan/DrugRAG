# src/data/chunkers/medical_section_chunker.py

from typing import List
from .base_chunker import BaseChunker
from .semantic_chunker import SemanticChunker
from src.data.loaders import DocumentChunk

class MedicalSectionChunker(BaseChunker):
    """
    医学专用切分器：
    - 对短字段（如 half_life, protein_binding）不切分
    - 对长字段（如 description, indications）使用 SemanticChunker
    """

    # 定义哪些字段可能很长，需要二次切分
    LONG_FIELDS = {
        "description",
        "indications",
        "pharmacodynamics",
        "mechanism_of_action",
        "toxicity",
        "metabolism",
        "absorption",
        "contraindications",
    }

    def __init__(self, max_chunk_size: int = 512, overlap: int = 50):
        self.semantic_chunker = SemanticChunker(max_chunk_size, overlap)

    def split(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        short_chunks = []
        long_chunks = []

        for chunk in chunks:
            field = chunk.metadata.get("field", "")
            if field in self.LONG_FIELDS:
                long_chunks.append(chunk)
            else:
                short_chunks.append(chunk)

        # 只对长字段进行语义切分
        refined_long_chunks = self.semantic_chunker.split(long_chunks) if long_chunks else []

        return short_chunks + refined_long_chunks