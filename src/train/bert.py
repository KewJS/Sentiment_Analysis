import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel
from transformers import BertTokenizer

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
    
    
class ReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    
    def __len__(self):
        return len(self.reviews)
    
    
    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(review,
                                              add_special_tokens=True,
                                              max_length=self.max_len,
                                              return_token_type_ids=False,
                                              pad_to_max_length=True,
                                              return_attention_mask=True,
                                              return_tensors="pt",
                                              )

        return {
            "review_text": review,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(target, dtype=torch.long)
            }


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = ReviewDataset(reviews=df.content.to_numpy(), targets=df.sentiment.to_numpy(), tokenizer=tokenizer, max_len=max_len)

    return DataLoader(ds, batch_size=batch_size, num_workers=4)