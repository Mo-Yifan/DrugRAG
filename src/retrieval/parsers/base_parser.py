# src/retrieval/parsers/base_parser.py

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from ..schemas import Drug


class BaseParser(ABC):
    """
    原始数据解析器基类
    输入：原始文件（XML/JSON等）
    输出：结构化 Drug 列表
    """

    @abstractmethod
    def parse(self, file_path: Path) -> List[Drug]:
        """
        解析输入文件，返回药物对象列表
        
        Args:
            file_path: 原始数据文件路径
            
        Returns:
            List[Drug]: 结构化药物数据
        """
        pass