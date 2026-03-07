# src/data/schemas.py

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from typing import List

class DrugMetadata(BaseModel):
    """
    DrugBank 药物块的标准化元数据
    """
    drugbank_id: str = Field(..., description="DrugBank 唯一标识符，如 DB00001")
    drug_name: str = Field(..., description="药物通用名")
    field: str = Field(..., description="来源字段名，如 'indications', 'half_life'")
    source: str = Field(default="DrugBank", description="数据来源")
    type: str = Field(default="small_molecule", description="药物类型")
    # 可扩展字段
    fda_approved: Optional[bool] = Field(None, description="是否 FDA 批准")
    indications_summary: Optional[str] = Field(None, description="适应症摘要（用于过滤）")
    synonyms: List[str] = Field(default_factory=list, description="药物别名列表")

class DocumentChunk(BaseModel):
    """
    标准化的文本块，用于 RAG 系统的输入单元
    """
    text: str = Field(..., min_length=1, description="文本内容")
    metadata: DrugMetadata = Field(..., description="结构化元数据")

    class Config:
        # 允许从字典实例化（方便与 LangChain / Chroma 集成）
        arbitrary_types_allowed = False
        json_encoders = {
            # 如有特殊类型可在此定义序列化方式
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，便于存储或日志"""
        return {
            "text": self.text,
            "metadata": self.metadata.dict()
        }