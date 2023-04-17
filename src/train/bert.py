import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer

class bert_classifier(nn.Module):
    def __init__(self, num_class=5, pretrain_model="bert-base-cased"):
        super(bert_classifier, self).__init__()
        self.pretrain_model = pretrain_model
        self.num_class = num_class

        self.bert = BertModel.from_pretrained(pretrain_model)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_class)
        
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_model)


    def forward(self, input_ids, attention_mask=None):
        ouput, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        drop_output = self.drop(pooled_output)
        linear_output = self.out(drop_output)
        return linear_output


    def get_num_class(self):
        return self.num_class


    def get_pretrain_model_name(self):
        return self.pretrain_model
    
    
    def get_tokenizer(self):
        return self.tokenizer
    