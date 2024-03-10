from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from config import Config
from transformers import BertTokenizer


class Labels:

    def __init__(self) -> None:
        self._labels = {
            0: "business",
            1: "entertainment",
            2: "sport",
            3: "tech",
            4: "politics",
        }

    def get(self):
        return self._labels

    def inverse(self):
        return {v: k for k, v in self._labels.items()}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df: Any) -> None:
        config = Config()
        tokenizer = BertTokenizer.from_pretrained(config.model.get("bert").get("name"))
        categories: Dict[str, int] = {
            "business": 0,
            "entertainment": 1,
            "sport": 2,
            "tech": 3,
            "politics": 4,
        }

        self._labels: List[int] = [categories[x] for x in df["category"]]
        self._texts: List[Dict[str, torch.Tensor]] = [
            tokenizer(
                text,
                padding="max_length",
                max_length=config.model.get("bert").get("length"),
                truncation=True,
                return_tensors="pt",
            )
            for text in df["text"]
        ]

    def classes(self) -> List[int]:
        return self._labels

    def __len__(self) -> int:
        return len(self._labels)

    def get_batch_labels(self, idx: int) -> np.ndarray:
        return np.array(self._labels[idx])

    def get_batch_texts(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._texts[idx]

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
        return self.get_batch_texts(idx), self.get_batch_labels(idx)
