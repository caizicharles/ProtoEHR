import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class MLP(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = None,
                 output_dim: int = None,
                 act_layer=nn.GELU,
                 dropout_prob: float = 0.):
        super().__init__()

        output_dim = output_dim or input_dim
        hidden_dim = hidden_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Triplet_Classifier(nn.Module):

    def __init__(self, args, model=None, config=None):
        super().__init__()

        self.bert_type = args['bert_type']
        self.freeze_bert = args['freeze_bert']
        self.input_dim = args['input_dim']
        self.hidden_dim = args['hidden_dim']
        self.out_dim = 1
        self.dropout = args['dropout']

        if self.bert_type == 'clinical':
            self.bert = model
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')

        if self.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.classifier = MLP(input_dim=self.input_dim,
                              hidden_dim=self.hidden_dim,
                              output_dim=self.out_dim,
                              dropout_prob=self.dropout)

    def forward(self, input_ids, attention_mask):

        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if self.bert_type != 'clinical':
            bert_out = bert_out.last_hidden_state[:, 0, :]

        out = self.classifier(bert_out)

        return out
