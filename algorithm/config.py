"""
配置文件
定义模型参数、标签体系等配置
"""
import os
import torch


class Config:
    """模型配置类"""

    def __init__(self):
        # 模型相关配置
        self.model_name = 'bert-base-chinese'
        self.max_len = 128
        self.dropout = 0.3  # 增加dropout防止过拟合

        # 训练相关配置
        self.batch_size = 16  # 减小batch size以获得更稳定的梯度
        self.epochs = 15  # 增加训练轮数
        self.lr = 3e-5  # 调整学习率
        self.warmup_ratio = 0.1  # warmup比例
        self.weight_decay = 0.01  # 权重衰减
        self.gradient_clip = 1.0  # 梯度裁剪

        # 设备配置
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 标签体系（BIO格式）
        # 注意: 根据数据分析,实际数据中没有中性标签,所以移除中性标签
        self.labels = [
            'O',
            'B-ASP', 'I-ASP',  # 维度词
            'B-SENT_POS', 'I-SENT_POS',  # 正向情感
            'B-SENT_NEG', 'I-SENT_NEG',  # 负向情感
        ]

        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.num_labels = len(self.labels)

        # 计算类别权重（用于处理数据不平衡）
        # 根据统计: O(89.8%), ASP(6%), SENT_NEG(2.67%), SENT_POS(1.47%)
        # 权重设置为频率的倒数
        self.label_weights = {
            'O': 0.1,  # 大幅降低O标签的权重
            'B-ASP': 2.0,
            'I-ASP': 2.0,
            'B-SENT_POS': 10.0,  # 正向情感权重最高
            'I-SENT_POS': 10.0,
            'B-SENT_NEG': 5.0,  # 负向情感权重中等
            'I-SENT_NEG': 5.0,
        }

        # 转换为tensor形式
        self.class_weights = torch.tensor(
            [self.label_weights[self.id2label[i]] for i in range(self.num_labels)],
            dtype=torch.float
        ).to(self.device)

        # 数据增强配置
        self.use_data_augmentation = True
        self.augment_positive_samples = True  # 对正向样本进行数据增强
        self.positive_oversample_ratio = 3  # 正向样本过采样倍数

        # 路径配置
        self.output_dir = '../saved_model'
        self.log_dir = 'logs'

        # 创建必要的目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def __str__(self):
        """打印配置信息"""
        config_str = "=" * 50 + "\n"
        config_str += "模型配置信息\n"
        config_str += "=" * 50 + "\n"
        config_str += f"模型: {self.model_name}\n"
        config_str += f"最大长度: {self.max_len}\n"
        config_str += f"Batch Size: {self.batch_size}\n"
        config_str += f"训练轮数: {self.epochs}\n"
        config_str += f"学习率: {self.lr}\n"
        config_str += f"设备: {self.device}\n"
        config_str += f"标签数量: {self.num_labels}\n"
        config_str += f"数据增强: {self.use_data_augmentation}\n"
        config_str += "=" * 50 + "\n"
        return config_str