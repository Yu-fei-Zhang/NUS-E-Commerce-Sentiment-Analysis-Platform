import os

import torch
from transformers import BertTokenizerFast

from model import BertCRFModel
from train import Config

config = Config()
os.makedirs(config.output_dir, exist_ok=True)

def extract_results(text, pred_tags, offset_mapping, id2label):
    results = []
    aspect_buffer = []
    sentiment_buffer = []
    current_aspect = None
    current_sentiment = None

    if not isinstance(pred_tags[0], list):
        pred_tags = [pred_tags]
    pred_tags_seq = pred_tags[0]

    offset_start_index = 0
    if offset_mapping.shape[0] > 0 and offset_mapping[0][0] == 0 and offset_mapping[0][1] > 0:
        offset_start_index = 1

    for i in range(offset_start_index, len(pred_tags_seq)):
        tag_id = pred_tags_seq[i]
        start, end = offset_mapping[i]

        if start == 0 and end == 0:
            continue
        if start >= len(text) or end > len(text):
            continue

        char = text[start:end]
        tag = id2label[tag_id] if 0 <= tag_id < len(id2label) else 'O'

        if tag.startswith('B-ASP'):
            if current_aspect:
                aspect_phrase = "".join(aspect_buffer)
                results.append({
                    'aspect_phrase': aspect_phrase,
                    'sentiment_phrase': None,
                    'sentiment': None,
                    'start': current_aspect[0],
                    'end': current_aspect[1]
                })
            aspect_buffer = [char]
            current_aspect = (start, end)
        elif tag.startswith('I-ASP') and current_aspect:
            aspect_buffer.append(char)
            current_aspect = (current_aspect[0], end)

        elif tag.startswith('B-SENT'):
            if current_sentiment:
                sentiment_phrase = "".join(sentiment_buffer)
                for res in reversed(results):
                    if res['sentiment'] is None:
                        res['sentiment_phrase'] = sentiment_phrase
                        res['sentiment'] = '正向' if 'POS' in tag else '负向' if 'NEG' in tag else '中性'
                        break
            sentiment_type = '正向' if 'POS' in tag else '负向' if 'NEG' in tag else '中性'
            sentiment_buffer = [char]
            current_sentiment = (start, end, sentiment_type)
        elif tag.startswith('I-SENT') and current_sentiment:
            curr_type = '正向' if 'POS' in tag else '负向' if 'NEG' in tag else '中性'
            if current_sentiment[2] == curr_type:  # 确保情感类型一致
                sentiment_buffer.append(char)
                current_sentiment = (current_sentiment[0], end, curr_type)

        elif tag == 'O':
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
                for res in reversed(results):
                    if res['sentiment'] is None:
                        res['sentiment_phrase'] = sentiment_phrase
                        res['sentiment'] = current_sentiment[2]
                        break
                current_sentiment = None
                sentiment_buffer = []

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
        for res in reversed(results):
            if res['sentiment'] is None:
                res['sentiment_phrase'] = sentiment_phrase
                res['sentiment'] = current_sentiment[2]
                break

    return results

def infer(text):
    model = BertCRFModel(config).to(config.device)
    model.load_state_dict(torch.load(os.path.join(config.output_dir, 'best_model.pt'), map_location=config.device))
    tokenizer = BertTokenizerFast.from_pretrained(config.model_name)
    model.eval()

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

    with torch.no_grad():
        pred_tags = model(input_ids, attention_mask)[0]  # 获取单个样本的预测

    results = extract_results(text, [pred_tags], offset_mapping, config.id2label)
    return results