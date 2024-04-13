import os
import re

import pandas as pd
import preprocessor as p
from config import Config


class PrepData:
    def __init__(self) -> None:
        config = Config()

        data_en = config.data.get("bbc-en")
        data_pt = config.data.get("bbc-pt")

        if not os.path.exists(data_en):
            raise FileNotFoundError(f"Dataset (1) not found: {data_en}")

        if not os.path.exists(data_pt):
            raise FileNotFoundError(f"Dataset (2) not found: {data_pt}")

        df_en = pd.read_csv(data_en)
        df_en["lang"] = 0

        df_pt = pd.read_csv(data_pt)
        df_pt["lang"] = 1

        df_multi = pd.concat([df_en, df_pt], ignore_index=True)
        df_multi["text"] = df_multi["text"].apply(self.preprocess_text)
        df_multi["text"] = df_multi["text"].str.lower().str.strip()

        df_multi = df_multi.sample(frac=1).reset_index(drop=True)

        multi = config.data.get("bbc-multi")
        df_multi.to_csv(multi, index=False, compression="gzip")
        print(f"Save multi-language dataset: {multi}")

    def valid(self) -> None:
        try:
            multi = Config().data.get("bbc-multi")

            if not os.path.exists(multi):
                raise FileNotFoundError(f"Dataset (1) not found: {multi}")

            df_multi = pd.read_csv(multi)
            lang = df_multi.groupby("lang").count()["text"].value_counts().to_list()[0]
            assert int(lang) == 2, "should contain only 2 languages"
            assert df_multi.shape[1] == 3, "should contain 3 columns only"

            category = df_multi.groupby("category").count().value_counts().to_list()
            assert sum(category) == 5, "should contain only 5 categories"

        except BaseException as e:
            print(f"Error: {str(e)}")

    @staticmethod
    def preprocess_text(sentence: str) -> str:
        sentence = p.clean(sentence)
        sentence = re.sub(r"http\S+", " ", sentence)
        sentence = re.sub(r"\s+", " ", sentence)
        sentence = re.sub(r"\|\|\|", " ", sentence)
        return str(sentence).lower()
