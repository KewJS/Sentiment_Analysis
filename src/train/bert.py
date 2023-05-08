import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, DistilBertModel

from src.config import Config


class BertClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(Config.MODELLING_CONFIG["PRE_TRAINED_MODEL_NAME"])
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)


    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
        output = self.drop(pooled_output)
        
        return self.out(output)
     