<img src="https://raw.githubusercontent.com/NeuroQuestAi/neuroquestai.github.io/main/brand/logo/neuroquest-orange-logo.png" align="right" width="65" height="65"/>

# Multi-Language Text Classification 🧠

[![Powered by NeuroQuestAI](https://img.shields.io/badge/powered%20by-NeuroQuestAI-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](
https://neuroquest.ai)
![python 3][python_version]
[![Code style: Black](https://img.shields.io/badge/code%20style-Black-000000.svg)](https://github.com/psf/black)
[![Packaged with Poetry][poetry-badge]](https://python-poetry.org/)

[poetry-badge]: https://img.shields.io/badge/packaging-poetry-cyan.svg
[python_version]: https://img.shields.io/static/v1.svg?label=python&message=3%20&color=blue

This project is a simple text classification 📝 using multi-language BERT. 🇬🇧 | 🇧🇷

### Project ☁️

Details of the **BERT** model used:

- [BERT multilingual base model (uncased)](https://huggingface.co/google-bert/bert-base-multilingual-uncased)

This project was inspired by this source: 

- [Documents Classification](https://www.kaggle.com/code/ouardasakram/documents-classification-using-bert-on-bbc-dataset/notebook)

`Note:` We reached **98**% accuracy.

### Requirements 🛠️

It is necessary:

- [Python 3.11](https://www.python.org/downloads/release/python-3110/)
- [Poetry](https://python-poetry.org/)
- [Git](https://git-scm.com/)

`Note:` A machine with a *GPU* is not required, but it is recommended to accelerate training.

### Datasets 📊

It was used the [BBC](https://www.kaggle.com/datasets/sainijagjit/bbc-dataset) dataset to classify texts into the following labels:

- Business 💼
- Entertainment 🎬
- Sport ⚽
- Tech 💻
- Politics 🏛️

The texts in the dataset were first translated into Brazilian Portuguese, using the [Google Translator API](https://cloud.google.com/translate/docs/reference/rest).
After that, the English 🇬🇧 and Portuguese 🇧🇷 texts were combined to create a multilingual version.

### Build and Running 🚀

Clone the project to your computer using Git and go to the project root folder:

```shell
git clone git@github.com:NeuroQuestAi/ml-text-classification.git && \
 cd ml-text-classification.git
```

Use poetry to access the project:

```shell
poetry shell
```

Install all dependencies:

```shell
poetry install && poetry update 
```

Run the model training and evaluation:

```shell
./train 
```

This will generate the torch model in the models folder. Then just test the predictions with the command:

```shell
./predictor 
```

`Note:` Model settings are in the [config.json](./config.json) file.

### Authors 👨‍💻

  * [Ederson Corbari](mailto:e@NeuralQuest.ai) 👽
