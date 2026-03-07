# src/data/loaders/drugbank_json_loader.py
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from .base_loader import BaseLoader
from src.retrieval.schemas import Drug

logger = logging.getLogger(__name__)

class DrugBankJSONLoader(BaseLoader):
    """
    专门解析 DrugBank 特殊 JSON 格式：
    结构为 { "SMILES": { "drugbank_id": "...", "name": "...", ... } }
    """

    def load(self, file_path: Path) -> List[Drug]:
        logger.info(f"📂 正在读取文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            logger.error("❌ 错误：根节点必须是字典 (Object)，当前格式不支持。")
            return []

        drugs = []
        total_items = len(data)
        logger.info(f"📦 检测到 {total_items} 个顶层条目 (SMILES -> DrugInfo)")

        success_count = 0
        skip_count = 0

        # 遍历字典的每一个 Key-Value 对
        # key = SMILES (我们暂时不需要，除非想存入 metadata)
        # value = 真正的药物信息字典
        for smiles, drug_info in data.items():
            if not isinstance(drug_info, dict):
                skip_count += 1
                continue

            drug = self._parse_item(drug_info, smiles)
            if drug:
                drugs.append(drug)
                success_count += 1
            else:
                skip_count += 1

            # 进度日志
            if (success_count + skip_count) % 1000 == 0:
                logger.info(f"🔄 已处理 {success_count + skip_count}/{total_items} ... 成功: {success_count}")

        logger.info(f"🎉 解析完成！成功: {success_count}, 跳过: {skip_count}")
        return drugs

    def _parse_item(self, item: Dict[str, Any], smiles: str = "") -> Optional[Drug]:
        try:
            # 1. 提取 ID (必须存在)
            drugbank_id = item.get("drugbank_id")
            if not drugbank_id:
                return None

            # 2. 提取名称
            name = item.get("name")
            if not name:
                name = f"Drug {drugbank_id}"  # 兜底

            # 3. 辅助函数：安全提取文本
            def get_text(*keys):
                for k in keys:
                    val = item.get(k)
                    if val:
                        if isinstance(val, list):
                            return " ".join(str(v) for v in val if v)
                        elif isinstance(val, dict):
                            # 处理某些字段可能是 {"value": "..."} 的情况
                            return str(val.get("value") or val.get("text") or val)
                        return str(val).strip()
                return None

            # 4. 提取并注入 Synonyms (别名)
            synonyms_raw = item.get("synonyms", [])

            # 处理 synonyms 可能是字典的情况 (如 {"synonym": [...]})
            if isinstance(synonyms_raw, dict):
                # 尝试常见键名
                synonyms_raw = synonyms_raw.get("synonym") or synonyms_raw.get("synonyms") or []

            # 确保是列表
            if not isinstance(synonyms_raw, list):
                synonyms_raw = [synonyms_raw] if synonyms_raw else []

            # 过滤空值并转为字符串列表
            clean_synonyms = [str(s).strip() for s in synonyms_raw if s and str(s).strip()]

            # 5. 提取临床字段
            description = get_text("description")
            indications = get_text("indication", "indications")  # 兼容单复数
            pharmacodynamics = get_text("pharmacology", "pharmacodynamics")  # 兼容不同键名
            mechanism = get_text("mechanism_of_action")
            toxicity = get_text("toxicity")
            metabolism = get_text("metabolism")
            half_life = get_text("half_life", "half-life")

            # 6. 提取 FDA 批准状态
            groups = item.get("groups", [])
            if isinstance(groups, list):
                fda_approved = "approved" in [str(g).lower() for g in groups]
            else:
                fda_approved = False

            # 7. 构建 Drug 对象
            drug = Drug(
                drugbank_id=str(drugbank_id),
                name=str(name),
                description=description,
                indications=indications,
                pharmacodynamics=pharmacodynamics,
                mechanism_of_action=mechanism,
                toxicity=toxicity,
                metabolism=metabolism,
                half_life=half_life,
                fda_approved=fda_approved,
                synonyms=clean_synonyms  # 👈 唯一新增的一行！
            )

            # 8. 可选：如果没有任何临床文本，可以选择跳过或保留
            # 这里我们只要有一点点文本就保留，避免过滤太严导致 0 数据
            if not drug.has_clinical_text():
                # 如果没有大段文本，至少要有名字和 ID，我们可以保留它（也许以后有用）
                # 如果你想严格过滤，可以取消下面这行的注释
                return None

            return drug
        except Exception as e:
            logger.error(f"解析单个药物失败: {e}")
            return None