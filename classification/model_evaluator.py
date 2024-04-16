import pandas as pd
import torch
from bert_classifier import BertClassifier
from dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


class ModelEvaluator:
    def __init__(self, model: BertClassifier) -> None:
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            self.model = self.model.cuda()

    def evaluate(self, df_test: pd.DataFrame) -> float:
        test_dataloader = DataLoader(Dataset(df_test), batch_size=2)

        total_acc_test = 0
        with torch.no_grad():

            for test_input, test_label in tqdm(test_dataloader, desc="Evaluating"):
                test_label = test_label.to(self.device)
                mask = test_input["attention_mask"].to(self.device)
                input_id = test_input["input_ids"].squeeze(1).to(self.device)

                output = self.model(input_id, mask)
                acc = (output.argmax(dim=1) == test_label).sum().item()
                total_acc_test += acc

        test_accuracy = total_acc_test / len(df_test)
        print(test_accuracy)
        print(f"Test Accuracy: {test_accuracy:.3f}")

        return test_accuracy
