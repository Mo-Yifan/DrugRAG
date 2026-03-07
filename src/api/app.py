# src/api/app.py

import logging
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# 导入路由器
from src.api.routes.rag_routes import router as rag_router

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DrugBank Clinical RAG",
    description="基于 DrugBank 知识库的临床药物问答系统",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册 API 路由
app.include_router(rag_router, prefix="/api", tags=["Clinical QA"])

# ✅ 关键修复：配置模板和静态文件目录
BASE_DIR = Path(__file__).resolve().parent
templates_path = BASE_DIR / "templates"
static_path = BASE_DIR / "static"

# 初始化 Jinja2 模板引擎
if templates_path.exists():
    templates = Jinja2Templates(directory=str(templates_path))
    logger.info(f"✅ 模板目录已加载：{templates_path}")
else:
    logger.error(f"❌ 模板目录不存在：{templates_path}")
    templates = None

# 挂载静态文件 (CSS, JS, images 等)
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    logger.info(f"✅ 静态文件目录已挂载：{static_path}")
else:
    logger.warning(f"⚠️ 静态文件目录不存在：{static_path}")

# ✅ 关键修复：添加根路径路由，返回 index.html
@app.get("/")
async def read_root(request: Request):
    if templates is None:
        return {"error": "Template engine not initialized. Please check logs."}
    
    # 渲染 templates/index.html
    # 注意：第一个参数是文件名，第二个参数必须包含 'request' 对象
    return templates.TemplateResponse("index.html", {"request": request})

# 健康检查接口
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    # 启动服务
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)