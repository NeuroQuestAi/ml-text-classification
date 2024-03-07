from typing import Any

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm


class ModelTrainer:
    def __init__(self, model: Any) -> None:
        print("Model Trainer")
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self,
        train_data: Any,
        val_data: Any,
        learning_rate: float,
        epochs: int,
        save_path: str,
    ) -> None:
        train_dataloader = DataLoader(train_data, batch_size=2, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=2)

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr=learning_rate)

        best_val_acc, best_epoch = (0, 0)

        self.model.to(self.device)
        criterion.to(self.device)

        for epoch_num in range(epochs):
            total_acc_train, total_loss_train = (0, 0)

            for train_input, train_label in tqdm(
                train_dataloader, desc=f"Epoch {epoch_num + 1}/{epochs}"
            ):
                train_label = train_label.to(self.device)
                mask = train_input["attention_mask"].to(self.device)
                input_id = train_input["input_ids"].squeeze(1).to(self.device)

                optimizer.zero_grad()
                output = self.model(input_id, mask)
                batch_loss = criterion(output, train_label.long())
                batch_loss.backward()
                optimizer.step()

                total_loss_train += batch_loss.item()
                total_acc_train += (output.argmax(dim=1) == train_label).sum().item()

            total_acc_val, total_loss_val = self.evaluate(val_dataloader, criterion)

            val_accuracy = total_acc_val / len(val_data)

            print(
                f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} | Train Accuracy: {total_acc_train / len(train_data): .3f} | Val Loss: {total_loss_val / len(val_data): .3f} | Val Accuracy: {val_accuracy: .3f}"
            )

            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_epoch = epoch_num + 1
                torch.save(self.model.state_dict(), save_path)

        print(
            f"Best validation accuracy: {best_val_acc} achieved at epoch {best_epoch}"
        )
