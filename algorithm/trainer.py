"""
训练模块
包含训练和评估函数
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, total_epochs):
    """
    训练一个epoch

    Args:
        model: 模型
        dataloader: 数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 设备
        epoch: 当前epoch
        total_epochs: 总epoch数

    Returns:
        平均损失
    """
    model.train()
    total_loss = 0

    # 使用tqdm显示进度条
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}")

    for step, batch in enumerate(progress_bar):
        # 梯度清零
        optimizer.zero_grad()

        # 数据移到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 前向传播
        loss = model(input_ids, attention_mask, labels=labels)

        # 反向传播
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新参数
        optimizer.step()
        scheduler.step()

        # 累计损失
        total_loss += loss.item()

        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / (step + 1):.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, id2label, device) -> Dict:
    """
    评估模型

    Args:
        model: 模型
        dataloader: 数据加载器
        id2label: ID到标签的映射
        device: 设备

    Returns:
        评估指标字典
    """
    model.eval()
    all_true_tags = []
    all_pred_tags = []

    print("\n开始评估...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 数据移到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()

            # 预测
            pred_tags_batch = model(input_ids, attention_mask)

            # 转换为标签序列
            for true_seq_ids, pred_seq_ids, mask in zip(
                    labels, pred_tags_batch, attention_mask.cpu().numpy()
            ):
                true_seq = []
                pred_seq = []

                # 获取有效长度（非padding部分）
                effective_length = mask.sum()

                # 遍历有效长度内的token
                for i in range(effective_length):
                    true_id = true_seq_ids[i]
                    pred_id = pred_seq_ids[i]

                    # 只处理非-100的标签
                    if true_id != -100:
                        # 确保ID有效
                        if 0 <= true_id < len(id2label):
                            true_tag = id2label[true_id]
                        else:
                            true_tag = 'O'

                        if 0 <= pred_id < len(id2label):
                            pred_tag = id2label[pred_id]
                        else:
                            pred_tag = 'O'

                        true_seq.append(true_tag)
                        pred_seq.append(pred_tag)

                # 添加到总列表
                if true_seq and pred_seq:
                    all_true_tags.append(true_seq)
                    all_pred_tags.append(pred_seq)

    # 检查是否有有效序列
    if not all_true_tags or not all_pred_tags:
        print("警告：没有有效的评估序列！")
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "report": {}
        }

    # 计算指标
    precision = precision_score(all_true_tags, all_pred_tags)
    recall = recall_score(all_true_tags, all_pred_tags)
    f1 = f1_score(all_true_tags, all_pred_tags)

    # 打印详细报告
    print("\n" + "=" * 60)
    print("评估结果:")
    print("=" * 60)
    report = classification_report(all_true_tags, all_pred_tags, digits=4)
    print(report)

    # 单独统计情感标签的性能
    print("\n情感标签性能:")
    print("-" * 60)
    sentiment_metrics = calculate_sentiment_metrics(all_true_tags, all_pred_tags)
    for key, value in sentiment_metrics.items():
        print(f"{key}: {value:.4f}")
    print("=" * 60)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": classification_report(all_true_tags, all_pred_tags, output_dict=True),
        "sentiment_metrics": sentiment_metrics
    }


def calculate_sentiment_metrics(true_tags: List[List[str]],
                                pred_tags: List[List[str]]) -> Dict[str, float]:
    """
    计算情感标签的专项指标

    Args:
        true_tags: 真实标签序列
        pred_tags: 预测标签序列

    Returns:
        情感标签的指标字典
    """
    # 提取情感标签
    sentiment_labels = ['B-SENT_POS', 'I-SENT_POS', 'B-SENT_NEG', 'I-SENT_NEG']

    true_sentiment = []
    pred_sentiment = []

    for true_seq, pred_seq in zip(true_tags, pred_tags):
        true_sent = [tag if tag in sentiment_labels else 'O' for tag in true_seq]
        pred_sent = [tag if tag in sentiment_labels else 'O' for tag in pred_seq]
        true_sentiment.append(true_sent)
        pred_sentiment.append(pred_sent)

    # 计算情感标签的指标
    pos_precision = precision_score([[t if 'POS' in t else 'O' for t in seq] for seq in true_sentiment],
                                    [[t if 'POS' in t else 'O' for t in seq] for seq in pred_sentiment])
    pos_recall = recall_score([[t if 'POS' in t else 'O' for t in seq] for seq in true_sentiment],
                              [[t if 'POS' in t else 'O' for t in seq] for seq in pred_sentiment])
    pos_f1 = f1_score([[t if 'POS' in t else 'O' for t in seq] for seq in true_sentiment],
                      [[t if 'POS' in t else 'O' for t in seq] for seq in pred_sentiment])

    neg_precision = precision_score([[t if 'NEG' in t else 'O' for t in seq] for seq in true_sentiment],
                                    [[t if 'NEG' in t else 'O' for t in seq] for seq in pred_sentiment])
    neg_recall = recall_score([[t if 'NEG' in t else 'O' for t in seq] for seq in true_sentiment],
                              [[t if 'NEG' in t else 'O' for t in seq] for seq in pred_sentiment])
    neg_f1 = f1_score([[t if 'NEG' in t else 'O' for t in seq] for seq in true_sentiment],
                      [[t if 'NEG' in t else 'O' for t in seq] for seq in pred_sentiment])

    return {
        'positive_precision': pos_precision,
        'positive_recall': pos_recall,
        'positive_f1': pos_f1,
        'negative_precision': neg_precision,
        'negative_recall': neg_recall,
        'negative_f1': neg_f1,
    }


def create_optimizer_and_scheduler(model, train_dataloader, config):
    """
    创建优化器和学习率调度器

    Args:
        model: 模型
        train_dataloader: 训练数据加载器
        config: 配置对象

    Returns:
        optimizer, scheduler
    """
    # 优化器
    # 对BERT和CRF使用不同的学习率
    bert_params = []
    crf_params = []
    other_params = []

    for name, param in model.named_parameters():
        if 'bert' in name:
            bert_params.append(param)
        elif 'crf' in name:
            crf_params.append(param)
        else:
            other_params.append(param)

    optimizer = optim.AdamW([
        {'params': bert_params, 'lr': config.lr},
        {'params': crf_params, 'lr': config.lr * 10},  # CRF使用更大的学习率
        {'params': other_params, 'lr': config.lr * 5}
    ], weight_decay=config.weight_decay)

    # 学习率调度器
    total_steps = len(train_dataloader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    print(f"优化器配置:")
    print(f"  - BERT参数学习率: {config.lr}")
    print(f"  - CRF参数学习率: {config.lr * 10}")
    print(f"  - 其他参数学习率: {config.lr * 5}")
    print(f"  - Warmup步数: {warmup_steps}")
    print(f"  - 总训练步数: {total_steps}")

    return optimizer, scheduler