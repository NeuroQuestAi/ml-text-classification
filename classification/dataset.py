import json
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from config import Config
from transformers import BertTokenizer


class Labels:
    def __init__(self, categories: List[str]) -> None:
        self._labels = {cat: i for i, cat in enumerate(categories)}

    def get(self) -> Dict[str, int]:
        return self._labels

    def inverse(self) -> Dict[int, str]:
        return {v: k for k, v in self._labels.items()}

    @staticmethod
    def save_to_file(categories: list) -> None:
        model_path = Config().model.get("results").get("models")
        labels_path = Config().model.get("results").get("model-labels")

        with open(f"{model_path}/{labels_path}", "w") as f:
            json.dump(categories, f)

    @staticmethod
    def get_from_file() -> dict:
        model_path = Config().model.get("results").get("models")
        labels_path = Config().model.get("results").get("model-labels")

        with open(f"{model_path}/{labels_path}", "r") as f:
            data = json.load(f)

        return data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df: Any) -> None:
        config = Config()
        tokenizer = BertTokenizer.from_pretrained(config.model.get("bert").get("name"))

        unique_categories = df["category"].unique().tolist()
        categories: Dict[str, int] = Labels(categories=unique_categories).get()

        Labels.save_to_file(categories=categories)

        self._labels: List[int] = [
            categories[str(x).strip().lower()] for x in df["category"]
        ]
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
