"""
数据处理模块
包含数据加载、预处理、数据增强和Dataset类定义
"""
import ast
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from collections import Counter


class AspectSentimentDataset(Dataset):
    """属性级情感分析数据集"""

    def __init__(self, texts, labels, tokenizer, max_len, label2id):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id
        self.valid_label_ids = set(self.label2id.values())

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        char_labels = self.labels[idx]

        # 确保文本非空
        if not isinstance(text, str) or not text.strip():
            text = "占位符"
            char_labels = ['O']

        # 确保标签是字符串列表
        if not isinstance(char_labels, list) or not all(isinstance(lbl, str) for lbl in char_labels):
            char_labels = ['O'] * len(text)

        # 确保标签长度与文本一致
        if len(char_labels) != len(text):
            if len(char_labels) < len(text):
                char_labels += ['O'] * (len(text) - len(char_labels))
            else:
                char_labels = char_labels[:len(text)]

        # Tokenize
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

        # 字符级标签 -> token级标签对齐
        token_labels = self._align_labels(char_labels, offset_mapping)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(token_labels, dtype=torch.long),
            'offset_mapping': offset_mapping,
            'text': text
        }

    def _align_labels(self, char_labels, offset_mapping):
        """将字符级标签对齐到token级"""
        token_labels = []

        for start, end in offset_mapping:
            # 特殊token (CLS, SEP, PAD)
            if start == 0 and end == 0:
                token_labels.append(-100)
            else:
                # 取token对应的第一个字符的标签
                if start < len(char_labels):
                    label = char_labels[start]
                    # 确保标签有效
                    token_label = self.label2id.get(label, self.label2id['O'])
                else:
                    token_label = self.label2id['O']

                token_labels.append(token_label)

        return token_labels


def load_and_preprocess_data(file_path: str, config) -> Tuple[List[str], List[List[str]]]:
    """
    加载并预处理数据

    Args:
        file_path: CSV文件路径
        config: 配置对象

    Returns:
        texts: 文本列表
        labels: 标签列表
    """
    print(f"正在加载数据: {file_path}")
    df = pd.read_csv(file_path)
    print(f"原始数据量: {len(df)}")

    # 过滤空值
    df = df.dropna(subset=['char_labels', 'content'])
    df = df[df['content'].str.strip().astype(bool)]
    print(f"删除空标签/文本后: {len(df)}")

    # 解析标签
    def safe_parse(x):
        try:
            return ast.literal_eval(x) if isinstance(x, str) else x
        except Exception:
            return None

    df['char_labels'] = df['char_labels'].apply(safe_parse)
    df = df[df['char_labels'].apply(lambda x: isinstance(x, list))]
    print(f"过滤无效标签后: {len(df)}")

    # 检查并修正标签
    all_labels = []
    for lbls in df['char_labels']:
        if isinstance(lbls, list):
            all_labels.extend(lbls)

    unique_labels = set(all_labels)
    invalid_labels = unique_labels - set(config.labels)
    if invalid_labels:
        print(f"警告：发现配置外的标签（将转换为'O'）：{invalid_labels}")

    # 文本转字符串并修正标签长度
    df['content'] = df['content'].astype(str)

    def fix_labels(row):
        """修正标签长度和有效性"""
        text = row['content']
        labels = row['char_labels']
        text_len = len(text)
        label_len = len(labels)

        # 调整长度
        if label_len < text_len:
            labels += ['O'] * (text_len - label_len)
        elif label_len > text_len:
            labels = labels[:text_len]

        # 确保所有标签有效
        row['char_labels'] = [lbl if lbl in config.labels else 'O' for lbl in labels]
        return row

    df = df.apply(fix_labels, axis=1)

    texts = df['content'].tolist()
    labels = df['char_labels'].tolist()

    print(f"最终有效数据量: {len(texts)}")

    # 统计标签分布
    print_label_statistics(labels)

    return texts, labels


def print_label_statistics(labels: List[List[str]]):
    """打印标签统计信息"""
    all_labels = []
    for label_seq in labels:
        all_labels.extend(label_seq)

    label_counts = Counter(all_labels)
    total = len(all_labels)

    print("\n标签分布:")
    print("-" * 40)
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        percentage = count / total * 100
        print(f"{label:15s}: {count:8d} ({percentage:5.2f}%)")
    print("-" * 40)


def oversample_positive_samples(texts: List[str],
                                labels: List[List[str]],
                                ratio: int = 3) -> Tuple[List[str], List[List[str]]]:
    """
    对包含正向情感的样本进行过采样

    Args:
        texts: 文本列表
        labels: 标签列表
        ratio: 过采样倍数

    Returns:
        增强后的文本和标签列表
    """
    positive_indices = []

    # 找出所有包含正向情感的样本
    for idx, label_seq in enumerate(labels):
        if any('SENT_POS' in label for label in label_seq):
            positive_indices.append(idx)

    print(f"\n找到 {len(positive_indices)} 个正向样本")
    print(f"将进行 {ratio}x 过采样")

    # 过采样
    augmented_texts = texts.copy()
    augmented_labels = labels.copy()

    for _ in range(ratio - 1):  # ratio-1 because we already have 1 copy
        for idx in positive_indices:
            augmented_texts.append(texts[idx])
            augmented_labels.append(labels[idx])

    print(f"过采样后数据量: {len(augmented_texts)}")

    # 打乱数据
    combined = list(zip(augmented_texts, augmented_labels))
    random.shuffle(combined)
    augmented_texts, augmented_labels = zip(*combined)

    return list(augmented_texts), list(augmented_labels)


def augment_text(text: str, labels: List[str], method: str = 'synonym') -> Tuple[str, List[str]]:
    """
    文本数据增强（简单版本）
    注意：这是一个简化版本，实际应用中可能需要更复杂的增强策略

    Args:
        text: 原始文本
        labels: 原始标签
        method: 增强方法

    Returns:
        增强后的文本和标签
    """
    # 这里可以实现各种增强方法
    # 如：同义词替换、回译、随机插入等
    # 简单起见，这里只返回原文本
    return text, labels