import os

import numpy as np
import pandas as pd
from bert_classifier import BertClassifier
from config import Config
from model_evaluator import ModelEvaluator
from model_trainer import ModelTrainer
from prep_data import PrepData

np.random.seed(Config().model.get("nn").get("seed"))


def train() -> None:
    config = Config()

    PrepData().valid()

    dir_model_path = config.model.get("results").get("models")
    save_model_path = config.model.get("results").get("model-multi")
    full_model_path = f"{dir_model_path}/{save_model_path}"
    dir_plots_path = config.model.get("results").get("plots")

    os.makedirs(dir_model_path, exist_ok=True)
    os.makedirs(dir_plots_path, exist_ok=True)

    df = pd.read_csv(config.data.get("bbc-multi"))

    print("DataFrame:")
    print(df.head(5))

    print("\nInfo:")
    print(df.info())

    print("\nCategories:")
    print(df.groupby(["category"]).size())

    print("\nLanguage:")
    print(df.groupby(["lang"]).size())

    df_train, df_val, df_test = np.split(
        df.sample(frac=1, random_state=42), [int(0.8 * len(df)), int(0.9 * len(df))]
    )

    print(
        f"\nDataFrame Split: -> Train: {len(df_train)} -> Val: {len(df_val)} -> Test: {len(df_test)}"
    )

    model = BertClassifier()
    trainer = ModelTrainer(model=model)

    lr = config.model.get("nn").get("lr")
    epochs = config.model.get("nn").get("epochs")

    print("BERT Setup:", config.model.get("bert"))
    print("Network Setup:", config.model.get("nn"))

    trainer.train(
        train_data=df_train,
        val_data=df_val,
        learning_rate=lr,
        epochs=epochs,
        save_path=full_model_path,
    )

    print("\nModel Evaluator")
    ModelEvaluator(model=model).evaluate(df_test=df_test)


def main() -> None:
    train()


if __name__ == "__main__":
    main()
