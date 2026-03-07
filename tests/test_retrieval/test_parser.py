# tests/test_retrieval/test_parser.py

import xml.etree.ElementTree as ET
from src.retrieval.parsers.drugbank_xml import DrugBankXMLParser


def test_drugbank_parser():
    """测试 XML 解析器（使用简化 XML）"""
    xml_content = """
    <drugbank>
        <drug type="biotech" created="2005-07-28" updated="2023-12-01">
            <drugbank-id primary="true">DB00945</drugbank-id>
            <name>Aspirin</name>
            <description>A common painkiller.</description>
            <indication>For pain and fever.</indication>
            <groups>
                <group>approved</group>
            </groups>
        </drug>
    </drugbank>
    """
    
    parser = DrugBankXMLParser()
    drugs = parser._parse_from_string(xml_content)
    
    assert len(drugs) == 1
    drug = drugs[0]
    assert drug.drugbank_id == "DB00945"
    assert drug.name == "Aspirin"
    assert drug.fda_approved is True