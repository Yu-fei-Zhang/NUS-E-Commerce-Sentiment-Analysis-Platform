def match_entity(token, entity_list):
    # 简单匹配：找包含token的实体（实际可用语义相似度匹配）
    for entity in entity_list:
        if token in entity:
            return entity
    return None  # 无匹配实体
# 示例：文本“续航强到让我惊讶”的token匹配实体
tokens = ["续航", "强", "到", "让", "我", "惊讶"]
entity_list = list(entity_to_id.keys())  # 知识图谱中的实体列表
matched_entities = [match_entity(token, entity_list) for token in tokens]
# 输出：["续航强", None, None, None, None, None]（假设“续航强”是实体）