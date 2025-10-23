import ast
import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report
import pandas as pd

from infer import extract_results
from model import BertCRFModel

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
class Config:
    def __init__(self):
        self.model_name = 'bert-base-chinese'
        self.max_len = 128
        self.batch_size = 32
        self.epochs = 1
        self.lr = 2e-5
        self.dropout = 0.1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 标签体系（BIO格式）
        self.labels = [
            'O',
            'B-ASP', 'I-ASP',  # 维度词
            'B-SENT_POS', 'I-SENT_POS',  # 正向情感
            'B-SENT_NEG', 'I-SENT_NEG',  # 负向情感
            'B-SENT_NEU', 'I-SENT_NEU'  # 中性情感
        ]
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.num_labels = len(self.labels)
        self.output_dir = 'saved_model'  # 模型保存路径


config = Config()
os.makedirs(config.output_dir, exist_ok=True)

class AspectSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, label2id):
        self.texts = texts
        self.labels = labels  # 字符级标签列表
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id
        self.valid_label_ids = set(self.label2id.values())

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        char_labels = self.labels[idx]

        if not isinstance(text, str) or not text.strip():
            text = "占位符"
            char_labels = ['O']

        if not isinstance(char_labels, list) or not all(isinstance(lbl, str) for lbl in char_labels):
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

        token_labels = []
        for start, end in offset_mapping:
            if start == 0 and end == 0:
                token_labels.append(-100)
            else:
                if start < len(char_labels):
                    label = char_labels[start]
                else:
                    label = 'O'
                token_label = self.label2id.get(label, self.label2id['O'])  # 无效标签默认'O'
                token_labels.append(token_label)

        labels_tensor = torch.tensor(token_labels, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels_tensor,
            'offset_mapping': offset_mapping,
            'text': text
        }

def load_data(file_path):
    df = pd.read_csv(file_path)

    df = df.dropna(subset=['char_labels', 'content'])
    df = df[df['content'].str.strip().astype(bool)]  # 过滤空文本

    # 解析标签（字符串转列表）
    def safe_parse(x):
        try:
            return ast.literal_eval(x) if isinstance(x, str) else x
        except Exception:
            return None

    df['char_labels'] = df['char_labels'].apply(safe_parse)
    df = df[df['char_labels'].apply(lambda x: isinstance(x, list))]

    all_labels = []
    for lbls in df['char_labels']:
        if isinstance(lbls, list):
            all_labels.extend(lbls)
    unique_labels = set(all_labels)
    invalid_labels = unique_labels - set(config.labels)

    df['content'] = df['content'].astype(str)

    def fix_length(row):
        text = row['content']
        labels = row['char_labels']
        text_len = len(text)
        label_len = len(labels)

        if label_len < text_len:
            labels += ['O'] * (text_len - label_len)
        elif label_len > text_len:
            labels = labels[:text_len]

        row['char_labels'] = [lbl if lbl in config.labels else 'O' for lbl in labels]
        return row

    df = df.apply(fix_length, axis=1)
    texts = df['content'].tolist()
    labels = df['char_labels'].tolist()

    return texts, labels

def main():
    texts, labels = load_data('/content/final_merged_dataset_fixed.csv')

    if len(texts) == 0:
        return

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    train_texts, train_labels = train_texts[:100], train_labels[:100]
    val_texts, val_labels = val_texts[:100], val_labels[:100]
    print(f"\n训练集大小: {len(train_texts)}")
    print(f"验证集大小: {len(val_texts)}")

    tokenizer = BertTokenizerFast.from_pretrained(config.model_name)

    train_dataset = AspectSentimentDataset(
        train_texts, train_labels, tokenizer, config.max_len, config.label2id
    )
    val_dataset = AspectSentimentDataset(
        val_texts, val_labels, tokenizer, config.max_len, config.label2id
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=0)

    model = BertCRFModel(config).to(config.device)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    best_f1 = 0.0
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, config.device)
        print(f"loss：{train_loss:.4f}")

        eval_report = evaluate(model, val_loader, config.id2label, config.device)
        if eval_report:
            current_f1 = eval_report['micro avg']['f1-score']
            if current_f1 > best_f1:
                best_f1 = current_f1
                torch.save(model.state_dict(), os.path.join(config.output_dir, 'best_model.pt'))
                print(f"save model（F1: {best_f1:.4f}）")
        else:
            print("fail to evaluate model.")


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        loss = model(input_ids, attention_mask, labels=labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, id2label, device):
    model.eval()
    all_true_tags = []
    all_pred_tags = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            texts = batch['text']
            offset_mappings = batch['offset_mapping']

            pred_tags = model(input_ids, attention_mask)

            for j, (true, pred, mask, text, offset_map) in enumerate(zip(
                labels, pred_tags, attention_mask.cpu().numpy(), texts, offset_mappings
            )):
                true_seq = []
                pred_seq = []
                true_len = len(true)
                pred_len = len(pred)

                for k in range(true_len):
                    t = true[k]
                    m = mask[k]

                    if m == 0 or t == -100:
                        continue

                    if k < pred_len:
                        p = pred[k]
                    else:
                        p = 0

                    true_tag = id2label[t] if (0 <= t < len(id2label)) else 'O'
                    pred_tag = id2label[p] if (0 <= p < len(id2label)) else 'O'

                    true_seq.append(true_tag)
                    pred_seq.append(pred_tag)

                if true_seq and pred_seq:
                    all_true_tags.append(true_seq)
                    all_pred_tags.append(pred_seq)

    if not all_true_tags or not all_pred_tags:
        return None

    report = classification_report(all_true_tags, all_pred_tags)
    print(report)

    if texts and len(pred_tags) > 0:
        sample_idx = 0
        sample_text = texts[sample_idx]
        sample_pred = pred_tags[sample_idx]
        sample_offset = offset_mappings[sample_idx].numpy()
        sample_results = extract_results(sample_text, [sample_pred], sample_offset, id2label)
        for res in sample_results:
            print(res)

    return classification_report(all_true_tags, all_pred_tags, output_dict=True)

if __name__ == "__main__":
    main()