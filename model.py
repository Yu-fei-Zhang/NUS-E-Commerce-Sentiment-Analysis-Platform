import ast
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from TorchCRF import CRF
from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report
import pandas as pd

class BertCRFModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout)
        self.hidden2tag = nn.Linear(self.bert.config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        logits = self.hidden2tag(self.dropout(last_hidden_state))

        crf_mask = attention_mask.bool()

        if labels is not None:
            valid_labels = labels.clone()
            valid_labels[valid_labels == -100] = 0
            max_valid_label = self.crf.num_tags - 1
            invalid_mask = (valid_labels < 0) | (valid_labels > max_valid_label)
            valid_labels[invalid_mask] = 0

            loss = -self.crf(logits, valid_labels, mask=crf_mask)
            return loss.mean()
        else:
            pred_tags = self.crf.decode(logits, mask=crf_mask)
            return pred_tags
    # 推理示例（训练完成后可单独运行）
    # sample_text = "这家店的衣服面料差，但版型很好"
    # print(infer(sample_text))