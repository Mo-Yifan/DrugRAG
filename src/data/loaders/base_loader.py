# src/data/loaders/base_loader.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pydantic import BaseModel

class DocumentChunk(BaseModel):
    """标准化的文档块，包含文本和元数据"""
    text: str
    metadata: Dict[str, Any]

class BaseLoader(ABC):
    """所有数据加载器的抽象基类"""

    @abstractmethod
    def load(self, file_path: str) -> List[DocumentChunk]:
        """
        从指定路径加载数据，并返回 DocumentChunk 列表
        """
        pass