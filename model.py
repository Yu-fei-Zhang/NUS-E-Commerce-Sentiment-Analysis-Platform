import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd


# --- 1. 配置与参数 ---
class Config:
    def __init__(self):
        self.model_name = 'bert-base-chinese'
        self.max_len = 128
        self.batch_size = 32
        self.epochs = 5
        self.lr = 2e-5
        self.dropout = 0.1
        self.num_aspect_labels = 6
        self.num_sentiment_labels = 3
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


config = Config()


# --- 2. 数据准备 ---
# 假设CSV包含三列：text, aspect_label, sentiment_label
class ReviewDataset(Dataset):
    def __init__(self, texts, aspect_labels, sentiment_labels, tokenizer, max_len):
        self.texts = texts
        self.aspect_labels = aspect_labels
        self.sentiment_labels = sentiment_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'aspect_label': torch.tensor(self.aspect_labels[idx], dtype=torch.long),
            'sentiment_label': torch.tensor(self.sentiment_labels[idx], dtype=torch.long)
        }


def load_data(file_path):
    df = pd.read_csv(file_path)
    texts = df['text'].tolist()
    aspect_labels = df['aspect_label'].tolist()
    sentiment_labels = df['sentiment_label'].tolist()
    return texts, aspect_labels, sentiment_labels


# --- 3. 模型定义 ---
class Bert_multitask(nn.Module):
    def __init__(self, bert_model_name, num_aspect_labels, num_sentiment_labels, dropout_rate):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout_rate)

        # 共享层
        self.shared_fc = nn.Linear(self.bert.config.hidden_size, 256)
        self.relu = nn.ReLU()

        # 维度分类头
        self.aspect_classifier = nn.Linear(256, num_aspect_labels)

        # 情感分类头
        self.sentiment_classifier = nn.Linear(256, num_sentiment_labels)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.pooler_output  # [CLS]向量

        # 共享特征处理
        x = self.dropout(cls_embedding)
        x = self.relu(self.shared_fc(x))

        # 两个任务分支
        aspect_logits = self.aspect_classifier(x)
        sentiment_logits = self.sentiment_classifier(x)

        return aspect_logits, sentiment_logits


# --- 4. 训练函数 ---
def train_epoch(model, dataloader, optimizer, scheduler, loss_fn_aspect, loss_fn_sentiment, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        aspect_labels = batch['aspect_label'].to(device)
        sentiment_labels = batch['sentiment_label'].to(device)

        aspect_logits, sentiment_logits = model(input_ids, attention_mask)

        loss_aspect = loss_fn_aspect(aspect_logits, aspect_labels)
        loss_sentiment = loss_fn_sentiment(sentiment_logits, sentiment_labels)

        # 多任务损失
        loss = 0.5 * loss_aspect + 0.5 * loss_sentiment

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, loss_fn_aspect, loss_fn_sentiment, device):
    model.eval()
    total_loss = 0
    all_aspect_preds = []
    all_sentiment_preds = []
    all_aspect_labels = []
    all_sentiment_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            aspect_labels = batch['aspect_label'].to(device)
            sentiment_labels = batch['sentiment_label'].to(device)

            aspect_logits, sentiment_logits = model(input_ids, attention_mask)

            loss_aspect = loss_fn_aspect(aspect_logits, aspect_labels)
            loss_sentiment = loss_fn_sentiment(sentiment_logits, sentiment_labels)
            loss = 0.5 * loss_aspect + 0.5 * loss_sentiment
            total_loss += loss.item()

            aspect_preds = torch.argmax(aspect_logits, dim=1)
            sentiment_preds = torch.argmax(sentiment_logits, dim=1)

            all_aspect_preds.extend(aspect_preds.cpu().numpy())
            all_sentiment_preds.extend(sentiment_preds.cpu().numpy())
            all_aspect_labels.extend(aspect_labels.cpu().numpy())
            all_sentiment_labels.extend(sentiment_labels.cpu().numpy())

    # 打印评估报告
    print("Aspect Classification Report:")
    print(classification_report(all_aspect_labels, all_aspect_preds))

    print("Sentiment Classification Report:")
    print(classification_report(all_sentiment_labels, all_sentiment_preds))

    return total_loss / len(dataloader)


# --- 5. 主训练流程 ---
def main():
    # 加载数据
    texts, aspect_labels, sentiment_labels = load_data('reviews.csv')
    train_texts, val_texts, train_aspect, val_aspect, train_sentiment, val_sentiment = train_test_split(
        texts, aspect_labels, sentiment_labels, test_size=0.2, random_state=42
    )

    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.model_name)

    # 创建数据集和数据加载器
    train_dataset = ReviewDataset(train_texts, train_aspect, train_sentiment, tokenizer, config.max_len)
    val_dataset = ReviewDataset(val_texts, val_aspect, val_sentiment, tokenizer, config.max_len)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # 初始化模型
    model = bert_multitask(
        bert_model_name=config.model_name,
        num_aspect_labels=config.num_aspect_labels,
        num_sentiment_labels=config.num_sentiment_labels,
        dropout_rate=config.dropout
    ).to(config.device)

    # 定义损失函数和优化器
    loss_fn_aspect = nn.CrossEntropyLoss()
    loss_fn_sentiment = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    # 学习率调度器
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(config.epochs):
        print(f"Epoch {epoch + 1}/{config.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn_aspect, loss_fn_sentiment,
                                 config.device)
        val_loss = evaluate(model, val_loader, loss_fn_aspect, loss_fn_sentiment, config.device)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print("Saved best model")


if __name__ == "__main__":
    main()