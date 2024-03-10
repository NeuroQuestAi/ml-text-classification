import torch
from bert_classifier import BertClassifier
from config import Config
from dataset import Labels
from prep_data import PrepData
from transformers import BertTokenizer


def model_inference():
    model = BertClassifier()

    dir_model = config.model.get("results").get("models")
    name_model = config.model.get("results").get("model-multi")

    model.load_state_dict(torch.load(f"{dir_model}/{name_model}"))
    model.eval()

    return model


def unseen_predict(config, device, model, sentence):
    tokenizer = BertTokenizer.from_pretrained(config.model.get("bert").get("name"))

    sentence_input = tokenizer(
        PrepData.preprocess_text(sentence=sentence),
        padding="max_length",
        max_length=config.model.get("bert").get("length"),
        truncation=True,
        return_tensors="pt",
    ).to(device)
    input_ids = sentence_input["input_ids"]
    mask = sentence_input["attention_mask"]

    with torch.no_grad():
        output = model(input_ids, mask)

    predicted_class_label = output.argmax(dim=1).item()
    predicted_class = Labels().get()[predicted_class_label]

    print(f"The predicted class is: {predicted_class}")


config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

unseen_predict(
    config=config,
    device=device,
    model=model_inference(),
    sentence="Manchester is a great football team in the history of sports.",
)
unseen_predict(
    config=config,
    device=device,
    model=model_inference(),
    sentence="Our job in politics is to help the population.",
)
