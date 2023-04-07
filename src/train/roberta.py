from transformers import RobertaModel
from transformers import RobertaTokenizer
import torch.nn as nn
import torch

class roberta_sum_classifier(nn.Module):
    def __init__(self, num_class=5, pretrain_model="roberta-base"):
        super(roberta_sum_classifier, self).__init__()
        self.pretrain_model = pretrain_model
        self.num_class = num_class

        self.roberta = RobertaModel.from_pretrained(pretrain_model)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.roberta.config.hidden_size, num_class)
        
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrain_model)


    def forward(self, input_ids, attention_mask=None):
        ouput = self.roberta(input_ids=input_ids, attention_mask=attention_mask)[0]
        drop_output = self.drop(ouput)
        linear_output = self.out(drop_output)
        p_sum = torch.mean(linear_output, 1)
        return p_sum


    def get_num_class(self):
        return self.num_class


    def get_pretrain_model_name(self):
        return self.pretrain_model
    
    
    def get_tokenizer(self):
        return self.tokenizer


class roberta_pool_classifier(nn.Module):
    def __init__(self, num_class=5, pretrain_model="roberta-base"):
        super(roberta_pool_classifier, self).__init__()
        self.pretrain_model = pretrain_model
        self.num_class = num_class

        self.roberta = RobertaModel.from_pretrained(pretrain_model)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.roberta.config.hidden_size, num_class)
        
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrain_model)


    def forward(self, input_ids, attention_mask=None):
        ouput, pooled_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        drop_output = self.drop(pooled_output)
        linear_output = self.out(drop_output)
        return linear_output


    def get_num_class(self):
        return self.num_class


    def get_pretrain_model_name(self):
        return self.pretrain_model
    
    
    def get_tokenizer(self):
        return self.tokenizer