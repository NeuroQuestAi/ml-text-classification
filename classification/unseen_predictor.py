from typing import Optional

import torch
from bert_classifier import BertClassifier
from config import Config
from dataset import Labels
from langdetect import detect
from prep_data import PrepData
from transformers import BertTokenizer


def model_inference() -> Optional[BertClassifier] | None:
    model = BertClassifier()

    dir_model = Config().model.get("results").get("models")
    name_model = Config().model.get("results").get("model-multi")

    try:
        model.load_state_dict(torch.load(f"{dir_model}/{name_model}"))
    except BaseException as e:
        print(f"Failed loading model: {str(e)}")
        return None

    return model


def unseen_predictor(model: BertClassifier, sentence: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(Config().model.get("bert").get("name"))

    sentence_input = tokenizer(
        PrepData.preprocess_text(sentence=sentence),
        padding="max_length",
        max_length=Config().model.get("bert").get("length"),
        truncation=True,
        return_tensors="pt",
    ).to(device)
    input_ids = sentence_input["input_ids"]
    mask = sentence_input["attention_mask"]

    with torch.no_grad():
        output = model(input_ids, mask)

    predicted_class_label = output.argmax(dim=1).item()
    predicted_class = Labels().inverse()[predicted_class_label]

    print(
        f"{sentence} -> {str(detect(sentence)).upper()} -> {str(predicted_class).upper()}"
    )


if __name__ == "__main__":
    model = model_inference()

    if model is None:
        exit(1)

    sentences = [
        "Os negócios são o tecido vital da economia, onde ideias se transformam em ações, e a inovação encontra seu caminho para o mercado. Empreendedores e empresas se unem para criar valor, atender às necessidades dos clientes e impulsionar o crescimento econômico.",
        "A variedade de formas de entretenimento reflete a diversidade de interesses e gostos das pessoas, proporcionando opções para todos os momentos e estados de espírito.",
        "Os valores como fair play, respeito e camaradagem são fundamentais no mundo esportivo, criando uma atmosfera de respeito mútuo e espírito esportivo. O esporte é muito mais do que uma competição; é uma fonte de inspiração, união e superação pessoal.",
        "Desde a revolução digital até as últimas descobertas científicas, a tecnologia molda nosso cotidiano e redefine o futuro. Ela amplia nossas capacidades, facilita a comunicação global, cria novas oportunidades de negócios e melhora a qualidade de vida das pessoas.",
        "A política reflete as diferentes visões, valores e interesses presentes em uma sociedade, e seu funcionamento democrático é fundamental para garantir a representatividade e a legitimidade das decisões tomadas.",
        "Businesses are the lifeblood of the economy, where ideas translate into actions and innovation finds its way to the market. Entrepreneurs and companies come together to create value, meet customer needs, and drive economic growth.",
        "From thrilling movies to engaging games, and soul-touching music, entertainment offers us moments of escapism and enjoyment.",
        "Sports are a universal passion that brings people from different backgrounds and cultures together in a healthy and thrilling competition. From major global events to local activities, sports are an expression of skill, teamwork, and overcoming challenges.",
        "Artificial intelligence, cloud computing, the Internet of Things, and other emerging technologies are shaping today's landscape and paving the way for even more significant advancements.",
        "Active citizen participation in political life, through voting, engagement in social movements, and monitoring government actions, is essential for strengthening democracy and ensuring accountability of elected leaders.",
    ]

    for sentence in sentences:
        unseen_predictor(model=model, sentence=sentence)
