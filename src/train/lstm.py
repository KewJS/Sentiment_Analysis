from transformers import RobertaModel
from transformers import RobertaTokenizer
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class LSTMClassifier(nn.Module):
    def __init__(self, no_layers, vocab_size, hidden_dim, embedding_dim, drop_prob=0.5):
        super(LSTMClassifier, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = vocab_size

        self.no_layers = no_layers
        self.vocab_size = vocab_size
        
        # # embedding
        self.out = nn.Embedding(vocab_size, embedding_dim)
        
        # # lstm
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim,
                            num_layers=no_layers, batch_first=True)
        
        # # dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_size, output_dim)
        self.sig = nn.Sigmoid()
        


    def forward(self, x, hidden):
        batch_size = x.size(0)
        
        # # embeddings and lstm_out
        embeds = self.embedding(x) # shape: B x S x Feature  since batch=True
        
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contigous().view(-1, self.hidden_dim)
        
        # # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        # # sigmoid function
        sig_out = self.sig(out)
        
        # # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get the last batch of labels
        
        return sig_out, hidden


    def init_hidden_size(self, batch_size):
        h0 = torch.zeros(self.no_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.no_layers, batch_size, self.hidden_dim).to(device)
        hidden = (h0, c0)
        
        return hidden