# src/api/routes/rag_routes.py

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging

from src.api.dependencies import get_rag_chain
from src.generation.rag_chain import RAGChain

logger = logging.getLogger(__name__)
router = APIRouter()

# ✅ 定义请求体模型，确保类型安全
class QueryRequest(BaseModel):
    question: str = Field(..., description="用户提出的问题")
    top_k: int = Field(default=3, description="检索相关的文档数量")
    fda_only: bool = Field(default=False, description="是否仅检索 FDA 批准的药物")

# ✅ 定义响应体模型（可选，但推荐用于文档生成）
class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    question: str

@router.post("/query", response_model=QueryResponse)
async def query_drugbank(
    request: QueryRequest,
    rag_chain: RAGChain = Depends(get_rag_chain)
):
    """
    临床药物问答接口
    -----------------
    接收用户问题，检索 DrugBank 知识库，并生成带引用的回答。
    """
    
    # 1. 安全检查
    if rag_chain is None:
        logger.error("❌ 致命错误：RAG Chain 未初始化 (None)")
        raise HTTPException(status_code=500, detail="RAG chain not initialized. Please check server logs.")

    try:
        logger.info(f"🔍 收到查询请求: [Question='{request.question}', TopK={request.top_k}, FDA_Only={request.fda_only}]")

        # 2. 执行 RAG 链路
        # 注意：如果 RAGChain.invoke 内部报错，这里会抛出异常并被下方的 except 捕获
        result = rag_chain.invoke(
            question=request.question,
            fda_only=request.fda_only
            # 如果需要动态调整 top_k，可以在 invoke 方法中增加该参数，这里暂时使用初始化时的默认值
        )

        # 3. 验证结果格式
        if not isinstance(result, dict) or "answer" not in result:
            logger.warning(f"⚠️ RAGChain 返回格式异常: {type(result)}")
            # 尝试兜底
            result = {
                "answer": str(result),
                "citations": [],
                "question": request.question
            }

        logger.info(f"✅ 查询成功，回答长度: {len(result['answer'])} chars")
        return result

    except HTTPException:
        # 重新抛出 HTTP 异常（不包装）
        raise
        
    except Exception as e:
        # ✅ 关键：记录完整的错误堆栈跟踪
        logger.error(f"❌ 处理查询时发生未捕获异常: {e}", exc_info=True)
        
        # 返回友好的错误信息给前端，同时包含具体错误原因
        error_detail = f"{type(e).__name__}: {str(e)}"
        raise HTTPException(
            status_code=500, 
            detail=f"查询失败: {error_detail}. 请查看服务器控制台获取详细堆栈信息。"
        )