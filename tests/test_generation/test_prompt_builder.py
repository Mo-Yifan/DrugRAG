# tests/test_generation/test_prompt_builder.py

from src.generation.prompt_builder import ClinicalQAPrompt


def test_prompt_construction():
    """测试提示词构建"""
    builder = ClinicalQAPrompt()
    
    question = "阿司匹林的作用机制是什么？"
    docs = [
        {"text": "Aspirin inhibits COX enzymes.", "metadata": {"drugbank_id": "DB00945"}},
        {"text": "It is an NSAID.", "metadata": {"drugbank_id": "DB00945"}}
    ]
    
    prompt = builder.build(question, docs)
    
    assert "阿司匹林的作用机制是什么？" in prompt
    assert "Aspirin inhibits COX enzymes." in prompt
    assert "<Source: DB00945>" in prompt
    assert "请用中文回答" in prompt  # 假设你要求中文输出