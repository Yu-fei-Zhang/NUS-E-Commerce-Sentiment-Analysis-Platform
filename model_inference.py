# ==================== 模型推理模块 ====================
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast
from TorchCRF import CRF
import os


class Config:
    """模型配置"""

    def __init__(self):
        self.model_name = 'bert-base-chinese'
        self.max_len = 128
        self.dropout = 0.1

        self.labels = [
            'O',
            'B-ASP', 'I-ASP',
            'B-SENT_POS', 'I-SENT_POS',
            'B-SENT_NEG', 'I-SENT_NEG',
            'B-SENT_NEU', 'I-SENT_NEU'
        ]
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.num_labels = len(self.labels)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = 'saved_model/best_model.pt'


class BertCRFModel(nn.Module):
    """BERT + CRF 模型"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout)
        self.hidden2tag = nn.Linear(self.bert.config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        emissions = self.hidden2tag(self.dropout(sequence_output))
        emissions = emissions.transpose(0, 1)

        if labels is not None:
            labels = labels.clone()
            labels[labels == -100] = 0
            labels = labels.transpose(0, 1)
            mask = attention_mask.bool().transpose(0, 1)
            log_likelihood = self.crf(emissions, labels, mask=mask)
            loss = -log_likelihood
            predictions = self.crf.decode(emissions, mask=mask)
            return loss, predictions
        else:
            mask = attention_mask.bool().transpose(0, 1)
            predictions = self.crf.decode(emissions, mask=mask)
            return predictions


class SentimentAnalyzer:
    """情感分析器"""

    def __init__(self, model_path=None):
        self.config = Config()
        if model_path:
            self.config.model_path = model_path

        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load model"""
        try:
            print(f"Loading tokenizer...")
            self.tokenizer = BertTokenizerFast.from_pretrained(self.config.model_name)

            print(f"Loading model: {self.config.model_path}")
            self.model = BertCRFModel(self.config).to(self.config.device)

            if os.path.exists(self.config.model_path):
                state_dict = torch.load(self.config.model_path, map_location=self.config.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print(f"✓ Model loaded successfully (device: {self.config.device})")
            else:
                print(f"⚠ Warning: Model file not found {self.config.model_path}")
                print("  Will use untrained model for inference")

        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise

    def predict(self, text):
        """Predict sentiment of single text"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded correctly")

        self.model.eval()

        # Tokenization
        encoding = self.tokenizer(
            text,
            max_length=self.config.max_len,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.config.device)
        attention_mask = encoding['attention_mask'].to(self.config.device)
        offset_mapping = encoding['offset_mapping'].squeeze(0).cpu().numpy()

        # 预测
        with torch.no_grad():
            pred_tags = self.model(input_ids, attention_mask)[0]

        # 解析实体和情感
        entities = []
        current_entity = None

        for idx, (start, end) in enumerate(offset_mapping):
            if start == end:
                continue

            if idx < len(pred_tags):
                tag_id = pred_tags[idx]
                if 0 <= tag_id < len(self.config.id2label):
                    tag = self.config.id2label[tag_id]

                    if tag.startswith('B-'):
                        if current_entity:
                            entities.append(current_entity)

                        entity_type = 'aspect' if 'ASP' in tag else 'sentiment'
                        sentiment = None
                        if 'POS' in tag:
                            sentiment = 'positive'
                        elif 'NEG' in tag:
                            sentiment = 'negative'
                        elif 'NEU' in tag:
                            sentiment = 'neutral'

                        current_entity = {
                            'text': text[start:end],
                            'start': int(start),
                            'end': int(end),
                            'type': entity_type,
                            'sentiment': sentiment
                        }
                    elif tag.startswith('I-') and current_entity:
                        current_entity['text'] += text[start:end]
                        current_entity['end'] = int(end)
                    elif tag == 'O' and current_entity:
                        entities.append(current_entity)
                        current_entity = None

        if current_entity:
            entities.append(current_entity)

        # 统计整体情感
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        for entity in entities:
            if entity['type'] == 'sentiment' and entity['sentiment']:
                sentiment_counts[entity['sentiment']] += 1

        # 打印调试信息(可选)
        # print(f"[DEBUG] Text: {text[:50]}...")
        # print(f"[DEBUG] Entities: {len(entities)}, Sentiment counts: {sentiment_counts}")

        # 确定整体情感 - 简化逻辑,更直接
        total_sentiments = sentiment_counts['positive'] + sentiment_counts['negative'] + sentiment_counts['neutral']

        if total_sentiments == 0:
            # 如果没有识别到任何情感词,返回中性
            overall_sentiment = 'neutral'
        else:
            # 简单多数投票 - 哪个最多选哪个
            if sentiment_counts['positive'] > sentiment_counts['negative'] and sentiment_counts['positive'] > \
                    sentiment_counts['neutral']:
                overall_sentiment = 'positive'
            elif sentiment_counts['negative'] > sentiment_counts['positive'] and sentiment_counts['negative'] > \
                    sentiment_counts['neutral']:
                overall_sentiment = 'negative'
            else:
                # 如果有并列或全部为0,返回中性
                overall_sentiment = 'neutral'

        return {
            'sentiment': overall_sentiment,
            'sentiment_counts': sentiment_counts,
            'entities': entities,
            'text': text
        }

    def predict_batch(self, texts):
        """批量预测"""
        results = []
        for text in texts:
            try:
                result = self.predict(text)
                results.append(result)
            except Exception as e:
                print(f"预测出错 (文本: {text[:50]}...): {e}")
                # 出错时返回中性结果
                results.append({
                    'sentiment': 'neutral',
                    'sentiment_counts': {'positive': 0, 'negative': 0, 'neutral': 0},
                    'entities': [],
                    'text': text
                })
        return results


# 全局分析器实例
_analyzer = None


def get_analyzer():
    """获取全局分析器实例"""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer


def analyze_text(text):
    """分析单个文本（便捷函数）"""
    analyzer = get_analyzer()
    return analyzer.predict(text)


def analyze_batch(texts):
    """批量分析（便捷函数）"""
    analyzer = get_analyzer()
    return analyzer.predict_batch(texts)
