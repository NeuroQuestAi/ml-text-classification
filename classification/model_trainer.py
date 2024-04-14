import os
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
from config import Config
from dataset import Dataset
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm


class ModelTrainer:
    def __init__(self, model: Any) -> None:
        self._config = Config()
        self._model = model
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._best_val_accuracy = float("-inf")
        self._train_losses, self._val_losses = [], []
        self._train_accuracies, self._val_accuracies = [], []

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        learning_rate: float,
        epochs: int,
        save_path: str,
    ):
        train, val = Dataset(train_data), Dataset(val_data)

        train_dataloader = DataLoader(train, batch_size=2, shuffle=True)
        val_dataloader = DataLoader(val, batch_size=2)

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self._model.parameters(), lr=learning_rate)

        self._model.to(self._device)
        criterion.to(self._device)

        for epoch_num in range(epochs):

            total_acc_train, total_loss_train = 0, 0

            for train_input, train_label in tqdm(train_dataloader, desc="Training"):

                train_label = train_label.to(self._device)
                mask = train_input["attention_mask"].to(self._device)
                input_id = train_input["input_ids"].squeeze(1).to(self._device)

                output = self._model(input_id, mask)

                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()

                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                self._model.zero_grad()
                batch_loss.backward()
                optimizer.step()

            total_acc_val, total_loss_val = 0, 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:
                    val_label = val_label.to(self._device)
                    mask = val_input["attention_mask"].to(self._device)
                    input_id = val_input["input_ids"].squeeze(1).to(self._device)

                    output = self._model(input_id, mask)

                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()

                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc

            avg_train_loss = total_loss_train / len(train_dataloader.dataset)
            avg_val_loss = total_loss_val / len(val_dataloader.dataset)
            avg_train_accuracy = total_acc_train / len(train_dataloader.dataset)
            avg_val_accuracy = total_acc_val / len(val_dataloader.dataset)

            self._train_losses.append(avg_train_loss)
            self._val_losses.append(avg_val_loss)
            self._train_accuracies.append(avg_train_accuracy)
            self._val_accuracies.append(avg_val_accuracy)

            print(
                f"Epochs: {epoch_num + 1} | Train Loss: {avg_train_loss:.3f} | Train Accuracy: {avg_train_accuracy:.3f} | Val Loss: {avg_val_loss:.3f} | Val Accuracy: {avg_val_accuracy:.3f}"
            )

            if avg_val_accuracy > self._best_val_accuracy:
                self._best_val_accuracy = avg_val_accuracy
                self.save_model(save_path=save_path)

        self.plot_training_progress(epochs=epochs)

    def save_model(self, save_path: str) -> None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        torch.save(self._model.state_dict(), save_path)
        print(f"Best model saved at: {save_path}")

    def plot_training_progress(self, epochs: int) -> None:
        dir_plots_path = self._config.model.get("results").get("plots")
        save_path = dir_plots_path

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epochs + 1), self._train_losses, label="Train Loss")
        plt.plot(range(1, epochs + 1), self._val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path + "/model-loss.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epochs + 1), self._train_accuracies, label="Train Accuracy")
        plt.plot(
            range(1, epochs + 1), self._val_accuracies, label="Validation Accuracy"
        )
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path + "/model-acc.png")
        plt.close()
