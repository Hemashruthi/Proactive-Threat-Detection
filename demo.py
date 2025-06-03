import json

with open("data/mitre_ttp_sentences.json") as f:
    examples = json.load(f)

max_char_len = max(len(e["sentence"]) for e in examples)
max_word_len = max(len(e["sentence"].split()) for e in examples)

print(f"🔠 Max character length: {max_char_len}")
print(f"🔤 Max word length: {max_word_len}")

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

max_token_len = 0
for ex in examples:
    tokens = tokenizer.encode(ex["sentence"], add_special_tokens=True)
    max_token_len = max(max_token_len, len(tokens))

print(f" Max tokenized length (BERT input): {max_token_len}")
