from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

example_text = "Tell me what it's like to be in the stars"
bert_input = tokenizer(
    example_text,
    padding="max_length",
    max_length=20,
    truncation=True,
    return_tensors="pt",
)

print(bert_input["input_ids"])
print(bert_input["token_type_ids"])
print(bert_input["attention_mask"])

target_text = tokenizer.decode(bert_input.input_ids[0])
print(target_text)
