import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

# ======================
# 数据集类
# ======================
class ABSADataset(Dataset):
    def __init__(self, path, tokenizer, max_len=64):
        self.data = pd.read_csv(path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['content']
        aspect = row['aspect_label']
        sentiment = int(row['sentiment_label'])

        encoded = self.tokenizer(
            f"{text} [SEP] {aspect}",
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': torch.tensor(sentiment)
        }

    def __len__(self):
        return len(self.data)


# ======================
# 模型定义
# ======================
class ABSAModel(nn.Module):
    def __init__(self, model_name='bert-base-chinese', num_labels=3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        logits = self.classifier(pooled)
        return logits


# ======================
# 训练与验证函数
# ======================
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    preds, labels_all = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids, attention_mask)
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    acc = accuracy_score(labels_all, preds)
    f1 = f1_score(labels_all, preds, average='macro')
    return acc, f1


# ======================
# 主函数
# ======================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    train_set = ABSADataset('data/train.csv', tokenizer)
    val_set = ABSADataset('data/val.csv', tokenizer)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    model = ABSAModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()

    best_f1 = 0.0
    save_dir = "best_model"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(3):
        print(f"\n===== Epoch {epoch+1} =====")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        acc, f1 = evaluate(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
            print(f"✅ New best model saved! F1={best_f1:.4f}")

    print("Training complete!")


if __name__ == "__main__":
    main()
