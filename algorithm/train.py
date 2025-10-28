"""
主训练脚本
用于训练属性级情感分析模型
"""
import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split
import random
import numpy as np

from config import Config
from data_utils import (
    load_and_preprocess_data,
    AspectSentimentDataset,
    oversample_positive_samples
)
from model import BertCRFModel
from trainer import train_epoch, evaluate, create_optimizer_and_scheduler
from inference import predict, format_results


def set_seed(seed: int = 42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"随机种子已设置为: {seed}")


def main():
    """主训练流程"""
    # 设置随机种子
    set_seed(42)

    # 加载配置
    config = Config()
    print(config)

    # 加载数据
    print("\n" + "=" * 80)
    print("步骤1: 加载数据")
    print("=" * 80)

    data_path = '../data/dataset_relabeled.csv'
    texts, labels = load_and_preprocess_data(data_path, config)

    if len(texts) == 0:
        print("错误: 没有有效数据！请检查数据文件。")
        return

    # 数据集划分
    print("\n" + "=" * 80)
    print("步骤2: 划分数据集")
    print("=" * 80)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    print(f"训练集大小: {len(train_texts)}")
    print(f"验证集大小: {len(val_texts)}")

    # 数据增强：对正向样本进行过采样
    if config.augment_positive_samples:
        print("\n" + "=" * 80)
        print("步骤3: 数据增强（正向样本过采样）")
        print("=" * 80)

        train_texts, train_labels = oversample_positive_samples(
            train_texts, train_labels, ratio=config.positive_oversample_ratio
        )
        print(f"增强后训练集大小: {len(train_texts)}")

    # 为了快速测试，可以限制数据量（实际训练时应该使用全部数据）
    # 注释掉下面这行以使用全部数据
    train_texts, train_labels = train_texts[:5000], train_labels[:5000]
    val_texts, val_labels = val_texts[:1000], val_labels[:1000]

    # 初始化tokenizer
    print("\n" + "=" * 80)
    print("步骤4: 初始化Tokenizer")
    print("=" * 80)

    tokenizer = BertTokenizerFast.from_pretrained(config.model_name)
    print(f"Tokenizer加载完成: {config.model_name}")

    # 创建数据集
    print("\n" + "=" * 80)
    print("步骤5: 创建数据集和数据加载器")
    print("=" * 80)

    train_dataset = AspectSentimentDataset(
        train_texts, train_labels, tokenizer, config.max_len, config.label2id
    )
    val_dataset = AspectSentimentDataset(
        val_texts, val_labels, tokenizer, config.max_len, config.label2id
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0  # Windows下设置为0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=0
    )

    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")

    # 初始化模型
    print("\n" + "=" * 80)
    print("步骤6: 初始化模型")
    print("=" * 80)

    model = BertCRFModel(config).to(config.device)
    print(f"模型已创建并移至设备: {config.device}")

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 创建优化器和调度器
    print("\n" + "=" * 80)
    print("步骤7: 创建优化器和学习率调度器")
    print("=" * 80)

    optimizer, scheduler = create_optimizer_and_scheduler(model, train_loader, config)

    # 训练循环
    print("\n" + "=" * 80)
    print("步骤8: 开始训练")
    print("=" * 80)

    best_f1 = 0.0
    best_epoch = 0
    patience = 3  # 早停的耐心值
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch}/{config.epochs}")
        print(f"{'=' * 80}")

        # 训练
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            config.device, epoch, config.epochs
        )
        print(f"\n训练损失: {train_loss:.4f}")

        # 评估
        eval_metrics = evaluate(model, val_loader, config.id2label, config.device)
        current_f1 = eval_metrics['f1']

        print(f"\n验证集F1: {current_f1:.4f}")
        print(f"验证集Precision: {eval_metrics['precision']:.4f}")
        print(f"验证集Recall: {eval_metrics['recall']:.4f}")

        # 保存最佳模型
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch
            patience_counter = 0

            # 保存模型
            save_path = os.path.join(config.output_dir, 'best_model')
            model.save_pretrained(save_path)
            print(f"\n✓ 新的最佳模型已保存! (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"\n当前F1未提升，耐心计数: {patience_counter}/{patience}")

        # 早停
        if patience_counter >= patience:
            print(f"\n早停触发！最佳F1: {best_f1:.4f} (Epoch {best_epoch})")
            break

    print("\n" + "=" * 80)
    print("训练完成!")
    print("=" * 80)
    print(f"最佳F1分数: {best_f1:.4f} (Epoch {best_epoch})")
    print(f"模型保存路径: {config.output_dir}/best_model")

    # 测试推理
    print("\n" + "=" * 80)
    print("步骤9: 测试推理")
    print("=" * 80)

    # 加载最佳模型
    best_model = BertCRFModel.load_pretrained(
        os.path.join(config.output_dir, 'best_model'),
        device=config.device
    )

    # 测试样例
    test_samples = [
        "这家店的衣服面料差，但版型很好",
        "手机屏幕很清晰，电池续航也不错",
        "酒店服务态度很好，环境也很舒适",
        "这本书内容很充实，但是价格有点贵",
        "餐厅的菜品味道一般，但是环境不错"
    ]

    print("\n测试样例预测结果:")
    for text in test_samples:
        results = predict(text, best_model, tokenizer, config)
        print(format_results(text, results))


if __name__ == "__main__":
    main()