from torch import nn
from transformers import BertModel
import torch


class BertClassifier(nn.Module):
    def __init__(self, dropout: float = 0.5) -> None:
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        return self.relu(linear_output)
