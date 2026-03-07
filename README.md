# DrugRAG：临床药物相互作用问答系统

一个基于 **DrugBank** 数据的检索增强生成（RAG）API，用于回答药物相互作用与药理学问题。  
设计原则：**准确、安全、可溯源**——绝不生成幻觉式医疗建议。

> ✅ 示例回答：  
> _“同时服用华法林和布洛芬可能会增加胃出血的风险。”_  
> 并附带 DrugBank 来源 ID（如 `<Source: DB00682, DB01050>`）。

---

## 🌐 重要提示：国内用户请配置 Hugging Face 镜像

本项目依赖 Hugging Face 的嵌入模型和重排序模型。**如果您在中国大陆**，必须在安装前设置镜像，否则模型下载将极慢或失败：

```bash
# 在终端中执行（或写入 ~/.bashrc）
export HF_ENDPOINT=https://hf-mirror.com
```

> 🔗 镜像站官网：[https://hf-mirror.com](https://hf-mirror.com)

---

## 📦 项目结构（GitHub 版本）

以下内容会上传至 GitHub，其余为本地生成文件：

```
├── data/
│   └── drugbank_small_molecule.json      # DrugBank 示例数据（小分子子集）
├── scripts/
│   ├── ingest.py                         # 从 DrugBank JSON 构建向量数据库
│   └── serve_api.py                      # 启动 FastAPI 服务
├── src/
│   ├── api/                              # FastAPI 应用与路由
│   ├── config/                           # 配置与路径管理
│   ├── data/                             # DrugBank 数据加载器与分块器
│   ├── generation/                       # LLM 推理链 + 临床问答 Prompt
│   └── retrieval/                        # 嵌入模型、检索器、重排序器
├── requirements.txt
└── README.md
```

> ❌ **不包含**：`.env`、`artifacts/`、`vectors/` 等本地生成目录（已加入 `.gitignore`）。

---

## ⚙️ 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 导入 DrugBank 数据（自动构建向量库）
```bash
python scripts/ingest.py
```

> 默认使用 `data/drugbank_small_molecule.json`。如需完整版，请自行提供合法授权的 DrugBank 数据。

### 3. 启动 API 服务
```bash
python scripts/serve_api.py
```

服务默认运行在 `http://localhost:8000`。

---

## 🧪 试用示例

查询药物组合安全性：
```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "氟西汀和曲马多能一起吃吗？"}'
```

返回结果包含：
- 基于 DrugBank 事实的 plain-language 风险提示
- 药物来源 ID（如 `<Source: DB00472, DB00691>`）

---

## 🔒 安全与设计原则

- **仅基于检索事实**：不引入外部知识，杜绝幻觉。
- **不提供医疗建议**：避免“请咨询医生”“注意监测”等非事实性表述。
- **机制感知推理**：能从“影响血小板”+“导致胃出血”推断出血风险。
- **多药独立检索**：确保每个药物的关键信息都被召回，避免漏检。

---

## 🛠️ 自定义扩展

- **切换大模型**：修改 `src/config/settings.py`，支持 OpenAI、本地 Llama 等。
- **接入新数据源**：扩展 `src/data/loaders/` 和 `src/retrieval/parsers/`。
- **优化检索效果**：调整分块策略（`medical_section_chunker.py`）或重排序模型（`bge_reranker.py`）。

---

## 📄 许可证

MIT 开源许可证。  
⚠️ **DrugBank 数据未包含在本仓库中**——您需自行提供合法授权的数据文件。