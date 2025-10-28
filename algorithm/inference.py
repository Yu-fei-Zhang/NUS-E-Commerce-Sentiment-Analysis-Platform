"""
推理和结果提取模块
包含模型推理和结构化结果提取功能
"""
import torch
from typing import List, Dict, Tuple


def extract_results(text: str,
                    pred_tags: List[int],
                    offset_mapping,
                    id2label: Dict[int, str]) -> List[Dict]:
    """
    从预测标签序列中提取结构化结果

    Args:
        text: 原始文本
        pred_tags: 预测的标签ID序列
        offset_mapping: token到字符的偏移映射
        id2label: ID到标签的映射

    Returns:
        结构化结果列表，每个元素包含：
        {
            'aspect_phrase': 属性词,
            'sentiment_phrase': 情感词,
            'sentiment': 情感极性（正向/负向/中性）,
            'start': 起始位置,
            'end': 结束位置
        }
    """
    results = []
    aspect_buffer = []
    sentiment_buffer = []
    current_aspect = None
    current_sentiment = None

    # 确保pred_tags是列表
    if isinstance(pred_tags, list) and len(pred_tags) > 0 and isinstance(pred_tags[0], list):
        pred_tags_seq = pred_tags[0]  # 取第一个序列
    else:
        pred_tags_seq = pred_tags

    # 确定有效长度（排除padding）
    effective_length = len(pred_tags_seq)
    for i in range(len(pred_tags_seq)):
        start, end = offset_mapping[i]
        if start == 0 and end == 0:  # padding token
            effective_length = i
            break

    # 跳过CLS token
    start_idx = 1 if effective_length > 0 and offset_mapping[0][0] == 0 and offset_mapping[0][1] > 0 else 0

    # 遍历有效token
    for i in range(start_idx, effective_length):
        tag_id = pred_tags_seq[i]
        start, end = offset_mapping[i]

        # 确保tag_id有效
        if tag_id < 0 or tag_id >= len(id2label):
            tag = 'O'
        else:
            tag = id2label[tag_id]

        # 确保offset在文本范围内
        if start >= len(text) or end > len(text):
            continue

        char = text[start:end]

        # 处理属性标签
        if tag.startswith('B-ASP'):
            # 保存之前的属性
            if current_aspect:
                aspect_phrase = "".join(aspect_buffer)
                results.append({
                    'aspect_phrase': aspect_phrase,
                    'sentiment_phrase': None,
                    'sentiment': None,
                    'start': current_aspect[0],
                    'end': current_aspect[1]
                })
            # 开始新属性
            aspect_buffer = [char]
            current_aspect = (start, end)

        elif tag.startswith('I-ASP') and current_aspect:
            # 继续当前属性
            aspect_buffer.append(char)
            current_aspect = (current_aspect[0], end)

        # 处理情感标签
        elif tag.startswith('B-SENT'):
            # 保存之前的情感
            if current_sentiment:
                sentiment_phrase = "".join(sentiment_buffer)
                _attach_sentiment_to_aspect(
                    results, sentiment_phrase, current_sentiment, start
                )

            # 开始新情感
            sentiment_type = _get_sentiment_type(tag)
            sentiment_buffer = [char]
            current_sentiment = (start, end, sentiment_type)

        elif tag.startswith('I-SENT') and current_sentiment:
            # 检查情感类型是否一致
            sentiment_type = _get_sentiment_type(tag)
            if sentiment_type == current_sentiment[2]:
                sentiment_buffer.append(char)
                current_sentiment = (current_sentiment[0], end, current_sentiment[2])

        # O标签或其他标签
        else:
            # 保存当前的属性和情感
            if current_aspect:
                aspect_phrase = "".join(aspect_buffer)
                results.append({
                    'aspect_phrase': aspect_phrase,
                    'sentiment_phrase': None,
                    'sentiment': None,
                    'start': current_aspect[0],
                    'end': current_aspect[1]
                })
                current_aspect = None
                aspect_buffer = []

            if current_sentiment:
                sentiment_phrase = "".join(sentiment_buffer)
                _attach_sentiment_to_aspect(
                    results, sentiment_phrase, current_sentiment, start
                )
                current_sentiment = None
                sentiment_buffer = []

    # 处理最后剩余的buffer
    if current_aspect:
        aspect_phrase = "".join(aspect_buffer)
        results.append({
            'aspect_phrase': aspect_phrase,
            'sentiment_phrase': None,
            'sentiment': None,
            'start': current_aspect[0],
            'end': current_aspect[1]
        })

    if current_sentiment:
        sentiment_phrase = "".join(sentiment_buffer)
        _attach_sentiment_to_aspect(
            results, sentiment_phrase, current_sentiment, len(text)
        )

    return results


def _get_sentiment_type(tag: str) -> str:
    """从标签中获取情感类型"""
    if 'POS' in tag:
        return '正向'
    elif 'NEG' in tag:
        return '负向'
    elif 'NEU' in tag:
        return '中性'
    else:
        return '未知'


def _attach_sentiment_to_aspect(results: List[Dict],
                                sentiment_phrase: str,
                                sentiment_info: Tuple[int, int, str],
                                current_pos: int):
    """
    将情感词关联到最近的属性词

    Args:
        results: 结果列表
        sentiment_phrase: 情感短语
        sentiment_info: (start, end, type)
        current_pos: 当前位置
    """
    # 向后查找最近的没有情感的属性
    found = False
    for res in reversed(results):
        if res['sentiment'] is None and res['start'] < current_pos:
            res['sentiment_phrase'] = sentiment_phrase
            res['sentiment'] = sentiment_info[2]
            found = True
            break

    # 如果没找到属性，创建一个独立的情感条目
    if not found:
        results.append({
            'aspect_phrase': None,
            'sentiment_phrase': sentiment_phrase,
            'sentiment': sentiment_info[2],
            'start': sentiment_info[0],
            'end': sentiment_info[1]
        })


def predict(text: str, model, tokenizer, config) -> List[Dict]:
    """
    对单个文本进行预测

    Args:
        text: 输入文本
        model: 模型
        tokenizer: 分词器
        config: 配置

    Returns:
        结构化结果列表
    """
    model.eval()

    # Tokenize
    encoding = tokenizer(
        text,
        max_length=config.max_len,
        padding='max_length',
        truncation=True,
        return_offsets_mapping=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(config.device)
    attention_mask = encoding['attention_mask'].to(config.device)
    offset_mapping = encoding['offset_mapping'].squeeze(0).numpy()

    # 预测
    with torch.no_grad():
        pred_tags = model(input_ids, attention_mask)

    # 提取结果
    results = extract_results(text, pred_tags, offset_mapping, config.id2label)

    return results


def batch_predict(texts: List[str], model, tokenizer, config, batch_size: int = 32) -> List[List[Dict]]:
    """
    批量预测

    Args:
        texts: 文本列表
        model: 模型
        tokenizer: 分词器
        config: 配置
        batch_size: 批大小

    Returns:
        每个文本的结构化结果列表
    """
    model.eval()
    all_results = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize
        encodings = tokenizer(
            batch_texts,
            max_length=config.max_len,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )

        input_ids = encodings['input_ids'].to(config.device)
        attention_mask = encodings['attention_mask'].to(config.device)
        offset_mappings = encodings['offset_mapping'].cpu().numpy()

        # 预测
        with torch.no_grad():
            pred_tags_batch = model(input_ids, attention_mask)

        # 提取每个文本的结果
        for text, pred_tags, offset_mapping in zip(batch_texts, pred_tags_batch, offset_mappings):
            results = extract_results(text, pred_tags, offset_mapping, config.id2label)
            all_results.append(results)

    return all_results


def format_results(text: str, results: List[Dict]) -> str:
    """
    格式化输出结果

    Args:
        text: 原始文本
        results: 结构化结果

    Returns:
        格式化的字符串
    """
    output = f"\n文本: {text}\n"
    output += "=" * 80 + "\n"

    if not results:
        output += "未识别到任何属性或情感\n"
    else:
        for i, res in enumerate(results, 1):
            output += f"\n结果 {i}:\n"
            output += "-" * 40 + "\n"

            if res['aspect_phrase']:
                output += f"  属性词: {res['aspect_phrase']}\n"

            if res['sentiment_phrase']:
                output += f"  情感词: {res['sentiment_phrase']}\n"
                output += f"  情感极性: {res['sentiment']}\n"

            output += f"  位置: [{res['start']}, {res['end']}]\n"

    output += "=" * 80 + "\n"
    return output