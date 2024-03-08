import torch
from config import Config
from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(
        self, dropout: float = Config().model.get("nn").get("dropout")
    ) -> None:
        super(BertClassifier, self).__init__()

        config = Config()
        self.bert = BertModel.from_pretrained(config.model.get("bert").get("name"))
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(
            config.model.get("bert").get("dimension"),
            config.model.get("bert").get("labels"),
        )
        self.relu = nn.ReLU()

    def forward(self, input_id: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        return self.relu(linear_output)
