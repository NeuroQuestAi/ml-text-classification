import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any


class ModelEvaluator:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self, test_data: Any) -> float:
        test_dataloader = DataLoader(test_data, batch_size=2)

        self.model.to(self.device)
        self.model.eval()

        total_acc_test = 0
        with torch.no_grad():
            for test_input, test_label in tqdm(test_dataloader, desc="Evaluating"):
                test_label = test_label.to(self.device)
                mask = test_input["attention_mask"].to(self.device)
                input_id = test_input["input_ids"].squeeze(1).to(self.device)

                output = self.model(input_id, mask)

                acc = (output.argmax(dim=1) == test_label).sum().item()
                total_acc_test += acc

        test_accuracy = total_acc_test / len(test_data)
        print(f"Test Accuracy: {test_accuracy:.3f}")

        return test_accuracy
