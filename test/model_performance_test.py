"""
Model Performance Test Tool
"""

import ast
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from TorchCRF import CRF
from transformers import BertTokenizerFast, BertModel
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import json
import tracemalloc
import psutil


import numpy as np


# ==================== 辅助函数 ====================
def convert_to_serializable(obj):
    """
    递归转换对象为JSON可序列化的类型
    处理 numpy 类型和其他不可序列化的对象
    """
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj


# ==================== 模型定义 ====================
class Config:
    def __init__(self):
        self.model_name = 'bert-base-chinese'
        self.max_len = 128
        self.batch_size = 32
        self.dropout = 0.1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        self.output_dir = '../saved_model/best_model'


class BertCRFModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout)
        # 使用 hidden2tag 以匹配训练时的模型结构
        self.hidden2tag = nn.Linear(self.bert.config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        logits = self.hidden2tag(self.dropout(last_hidden_state))

        # Create CRF mask based on attention_mask
        crf_mask = attention_mask.bool()

        if labels is not None:
            # Create a version of labels where -100 is replaced with a valid tag index
            valid_labels = labels.clone()
            valid_labels[valid_labels == -100] = 0  # Replace -100 with 'O' tag index

            # Calculate CRF loss
            loss = -self.crf(logits, valid_labels, mask=crf_mask)
            predictions = self.crf.decode(logits, mask=crf_mask)
            return loss.mean(), predictions
        else:
            # For decoding
            pred_tags = self.crf.decode(logits, mask=crf_mask)
            return pred_tags


class AspectSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, label2id):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        char_labels = self.labels[idx]

        if not isinstance(text, str) or not text.strip():
            text = "占位符"
            char_labels = ['O']

        if not isinstance(char_labels, list):
            char_labels = ['O'] * len(text)

        if len(char_labels) != len(text):
            if len(char_labels) < len(text):
                char_labels += ['O'] * (len(text) - len(char_labels))
            else:
                char_labels = char_labels[:len(text)]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        offset_mapping = encoding['offset_mapping'].squeeze(0).numpy()

        token_labels = [-100] * len(input_ids)
        for token_idx, (start, end) in enumerate(offset_mapping):
            if start == 0 and end == 0:
                token_labels[token_idx] = -100
            else:
                if start < len(char_labels):
                    label = char_labels[start]
                else:
                    label = 'O'
                token_label = self.label2id.get(label, self.label2id['O'])
                token_labels[token_idx] = token_label

        labels_tensor = torch.tensor(token_labels, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels_tensor,
            'text': text
        }


# ==================== 性能测试类 ====================
class ModelPerformanceTester:
    def __init__(self, model_path: str, config: Config):
        self.config = config
        self.device = config.device
        self.tokenizer = BertTokenizerFast.from_pretrained(config.model_name)

        print(f"正在加载模型: {model_path}")
        print(f"使用设备: {self.device}")

        self.model = BertCRFModel(config).to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("✓ 模型加载成功")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        self.model.eval()

    def test_inference_speed(self, test_texts: List[str], num_runs: int = 100) -> Dict:
        """测试推理速度"""
        print("\n" + "="*60)
        print("1. 推理速度测试")
        print("="*60)

        results = {
            'single_inference': [],
            'batch_inference': []
        }

        # 单样本推理速度测试
        print("\n[1.1] 单样本推理速度测试")
        test_text = test_texts[0] if test_texts else "这家店的衣服面料差，但版型很好"

        # 预热
        for _ in range(10):
            self._infer_single(test_text)

        # 正式测试
        times = []
        for i in range(num_runs):
            start = time.time()
            self._infer_single(test_text)
            end = time.time()
            times.append((end - start) * 1000)  # 转换为毫秒

        results['single_inference'] = {
            'mean_ms': np.mean(times),
            'median_ms': np.median(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'p95_ms': np.percentile(times, 95),
            'p99_ms': np.percentile(times, 99)
        }

        print(f"  平均推理时间: {results['single_inference']['mean_ms']:.2f} ms")
        print(f"  中位数时间: {results['single_inference']['median_ms']:.2f} ms")
        print(f"  95分位数: {results['single_inference']['p95_ms']:.2f} ms")
        print(f"  99分位数: {results['single_inference']['p99_ms']:.2f} ms")

        # 批量推理速度测试
        if len(test_texts) >= 32:
            print("\n[1.2] 批量推理速度测试 (batch_size=32)")
            batch_texts = test_texts[:32]

            # 预热
            for _ in range(5):
                self._infer_batch(batch_texts)

            # 正式测试
            batch_times = []
            for i in range(20):
                start = time.time()
                self._infer_batch(batch_texts)
                end = time.time()
                batch_times.append((end - start) * 1000)

            results['batch_inference'] = {
                'mean_ms': np.mean(batch_times),
                'median_ms': np.median(batch_times),
                'throughput': 32 / (np.mean(batch_times) / 1000)  # 样本/秒
            }

            print(f"  批量平均时间: {results['batch_inference']['mean_ms']:.2f} ms")
            print(f"  吞吐量: {results['batch_inference']['throughput']:.2f} 样本/秒")

        return results

    def test_accuracy(self, test_data_path: str = None,
                     test_texts: List[str] = None,
                     test_labels: List[List[str]] = None) -> Dict:
        """测试模型准确率"""
        print("\n" + "="*60)
        print("2. 准确率测试")
        print("="*60)

        if test_data_path:
            print(f"\n从文件加载测试数据: {test_data_path}")
            test_texts, test_labels = self._load_test_data(test_data_path)
        elif test_texts and test_labels:
            print(f"\n使用提供的测试数据: {len(test_texts)} 条")
        else:
            print("\n未提供测试数据，跳过准确率测试")
            return {}

        # 创建测试数据集
        test_dataset = AspectSentimentDataset(
            test_texts, test_labels, self.tokenizer,
            self.config.max_len, self.config.label2id
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 评估
        all_true_tags = []
        all_pred_tags = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']

                predictions = self.model(input_ids, attention_mask)

                for j in range(len(labels)):
                    true_seq_ids = labels[j].cpu().numpy()
                    pred_seq_ids = predictions[j]

                    true_seq = []
                    pred_seq = []

                    effective_length = min(len(true_seq_ids), len(pred_seq_ids))
                    for k in range(effective_length):
                        true_id = true_seq_ids[k]
                        pred_id = pred_seq_ids[k]

                        if true_id != -100:
                            if 0 <= true_id < len(self.config.id2label):
                                true_tag = self.config.id2label[true_id]
                            else:
                                true_tag = 'O'
                            true_seq.append(true_tag)

                            if 0 <= pred_id < len(self.config.id2label):
                                pred_tag = self.config.id2label[pred_id]
                            else:
                                pred_tag = 'O'
                            pred_seq.append(pred_tag)

                    if true_seq and pred_seq:
                        all_true_tags.append(true_seq)
                        all_pred_tags.append(pred_seq)

        if not all_true_tags:
            print("警告: 没有有效的测试数据")
            return {}

        # 计算指标
        print("\n[2.1] 整体性能指标:")
        report = classification_report(all_true_tags, all_pred_tags, output_dict=True)

        metrics = {
            'micro_f1': f1_score(all_true_tags, all_pred_tags),
            'precision': precision_score(all_true_tags, all_pred_tags),
            'recall': recall_score(all_true_tags, all_pred_tags)
        }

        print(f"  Micro F1: {metrics['micro_f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")

        # 各类别性能
        print("\n[2.2] 各类别详细指标:")
        print(classification_report(all_true_tags, all_pred_tags))

        return {
            'metrics': metrics,
            'detailed_report': report
        }

    def test_memory_usage(self, test_texts: List[str]) -> Dict:
        """测试内存使用"""
        print("\n" + "="*60)
        print("3. 内存使用测试")
        print("="*60)

        process = psutil.Process()

        # 获取模型参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\n[3.1] 模型参数量:")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")

        # 测试推理内存
        print(f"\n[3.2] 推理内存占用:")
        tracemalloc.start()
        mem_before = process.memory_info().rss / 1024 / 1024

        # 执行推理
        test_text = test_texts[0] if test_texts else "这家店的衣服面料差，但版型很好"
        for _ in range(10):
            self._infer_single(test_text)

        mem_after = process.memory_info().rss / 1024 / 1024
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"  推理前内存: {mem_before:.2f} MB")
        print(f"  推理后内存: {mem_after:.2f} MB")
        print(f"  内存增量: {mem_after - mem_before:.2f} MB")
        print(f"  峰值内存: {peak / 1024 / 1024:.2f} MB")

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,
            'inference_memory_mb': mem_after - mem_before,
            'peak_memory_mb': peak / 1024 / 1024
        }

    def test_robustness(self, base_text: str = "这家店的衣服面料差，但版型很好") -> Dict:
        """测试模型鲁棒性"""
        print("\n" + "="*60)
        print("4. 鲁棒性测试")
        print("="*60)

        test_cases = {
            '原始文本': base_text,
            '短文本': "很好",
            '长文本': base_text * 5,
            '特殊字符': base_text + "!@#$%^&*()",
            '数字混合': base_text + "123456",
            '英文混合': base_text + "very good",
            '空格文本': "  " + base_text + "  ",
            '重复文本': "商品质量好好好好好非常好"

        }

        results = {}
        print("\n[4.1] 不同输入测试:")

        for name, text in test_cases.items():
            try:
                start = time.time()
                result = self._infer_single(text)
                elapsed = (time.time() - start) * 1000

                # 提取情感结果
                aspects = [r for r in result if r['type'] == 'aspect']
                sentiments = [r for r in result if r['type'] == 'sentiment']

                results[name] = {
                    'success': True,
                    'time_ms': elapsed,
                    'aspects_found': len(aspects),
                    'sentiments_found': len(sentiments),
                    'text_length': len(text)
                }
                print(f"  ✓ {name}: {elapsed:.2f}ms | 方面:{len(aspects)} 情感:{len(sentiments)}")
            except Exception as e:
                results[name] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"  ✗ {name}: 失败 - {str(e)}")

        return results

    def test_edge_cases(self) -> Dict:
        """测试边界情况"""
        print("\n" + "="*60)
        print("5. 边界情况测试")
        print("="*60)

        edge_cases = {
            '空文本': "",
            '单字': "好",
            '纯标点': "。。。！！！",
            '超长文本': "商品质量" + "非常" * 100 + "好",
            '纯数字': "12345678",
            '纯英文': "very good quality",
            '特殊Unicode': "😊商品质量👍🎉",
            '换行文本': "第一行\n第二行商品质量很好\n第三行"
        }

        results = {}
        print("\n[5.1] 边界情况测试:")

        for name, text in edge_cases.items():
            try:
                result = self._infer_single(text)
                results[name] = {
                    'success': True,
                    'result_count': len(result)
                }
                print(f"  ✓ {name}: 成功 (结果数: {len(result)})")
            except Exception as e:
                results[name] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"  ✗ {name}: {str(e)[:50]}")

        return results

    def _infer_single(self, text: str) -> List[Dict]:
        """单样本推理"""
        encoding = self.tokenizer(
            text,
            max_length=self.config.max_len,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        offset_mapping = encoding['offset_mapping'].squeeze(0).cpu().numpy()

        with torch.no_grad():
            pred_tags = self.model(input_ids, attention_mask)[0]

        # 提取结果
        results = []
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
                            results.append(current_entity)

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
                        results.append(current_entity)
                        current_entity = None

        if current_entity:
            results.append(current_entity)

        return results

    def _infer_batch(self, texts: List[str]):
        """批量推理"""
        encodings = self.tokenizer(
            texts,
            max_length=self.config.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)

        with torch.no_grad():
            self.model(input_ids, attention_mask)

    def _load_test_data(self, file_path: str) -> Tuple[List[str], List[List[str]]]:
        """加载测试数据"""
        df = pd.read_csv(file_path)
        df = df.dropna(subset=['char_labels', 'content'])
        df = df[df['content'].str.strip().astype(bool)]

        def safe_parse(x):
            try:
                return ast.literal_eval(x) if isinstance(x, str) else x
            except:
                return None

        df['char_labels'] = df['char_labels'].apply(safe_parse)
        df = df[df['char_labels'].apply(lambda x: isinstance(x, list))]
        df['content'] = df['content'].astype(str)

        texts = df['content'].tolist()
        labels = df['char_labels'].tolist()

        return texts, labels

    def run_all_tests(self, test_data_path: str = None,
                      test_texts: List[str] = None,
                      test_labels: List[List[str]] = None,
                      save_report: bool = True) -> Dict:
        """运行所有测试"""
        print("\n" + "="*80)
        print(" "*20 + "模型性能综合测试报告")
        print("="*80)

        all_results = {}

        # 准备测试文本
        if not test_texts:
            test_texts = [
                "这家店的衣服面料差，但版型很好",
                "服务态度很好，环境也不错",
                "价格太贵了，性价比不高",
                "菜品味道一般，但是分量很足"
            ] * 10  # 创建足够的测试数据

        # 1. 推理速度测试
        all_results['inference_speed'] = self.test_inference_speed(test_texts)

        # 2. 准确率测试
        if test_data_path or (test_texts and test_labels):
            all_results['accuracy'] = self.test_accuracy(test_data_path, test_texts, test_labels)

        # 3. 内存使用测试
        all_results['memory'] = self.test_memory_usage(test_texts)

        # 4. 鲁棒性测试
        all_results['robustness'] = self.test_robustness()

        # 5. 边界情况测试
        all_results['edge_cases'] = self.test_edge_cases()

        # 生成总结
        print("\n" + "="*80)
        print(" "*30 + "测试总结")
        print("="*80)

        if 'inference_speed' in all_results and 'single_inference' in all_results['inference_speed']:
            speed = all_results['inference_speed']['single_inference']
            print(f"\n✓ 推理速度: 平均 {speed['mean_ms']:.2f} ms/样本")

        if 'accuracy' in all_results and 'metrics' in all_results['accuracy']:
            acc = all_results['accuracy']['metrics']
            print(f"✓ 模型准确率: F1={acc['micro_f1']:.4f}, P={acc['precision']:.4f}, R={acc['recall']:.4f}")

        if 'memory' in all_results:
            mem = all_results['memory']
            print(f"✓ 内存使用: 模型{mem['model_size_mb']:.2f}MB, 推理峰值{mem['peak_memory_mb']:.2f}MB")

        if 'robustness' in all_results:
            robust = all_results['robustness']
            success_count = sum(1 for r in robust.values() if r.get('success', False))
            print(f"✓ 鲁棒性: {success_count}/{len(robust)} 测试通过")

        if 'edge_cases' in all_results:
            edge = all_results['edge_cases']
            success_count = sum(1 for r in edge.values() if r.get('success', False))
            print(f"✓ 边界测试: {success_count}/{len(edge)} 测试通过")

        print("\n" + "="*80)

        # 保存报告
        if save_report:
            report_path = 'performance_test_report.json'  # 使用相对路径，保存在当前目录
            # 转换为可序列化的格式
            serializable_results = convert_to_serializable(all_results)
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            print(f"\n详细报告已保存至: {report_path}")

        return all_results


# ==================== 主函数 ====================
def main():
    # 配置
    config = Config()
    model_path = os.path.join(config.output_dir, 'previous_model.pt')

    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先训练模型或指定正确的模型路径")
        return

    # 创建测试器
    tester = ModelPerformanceTester(model_path, config)

    # 运行测试 (可以提供测试数据路径)
    test_data_path = '../data/test_dataset.csv'  # 如有测试数据
    results = tester.run_all_tests(
        test_data_path=test_data_path,  # 取消注释以使用测试数据
        save_report=True
    )

    return results


if __name__ == "__main__":
    main()