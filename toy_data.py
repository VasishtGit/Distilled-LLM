from datasets import Dataset
from transformers import AutoTokenizer

texts = [
"AI will change the future of humanity.",
"Knowledge distillation makes small models smarter.",
"Kalman filters smooth noisy signals.",
"Transformers are powerful sequence models.",
"Small models are faster and lighter to run.",
] 


tok = AutoTokenizer.from_pretrained("gpt2")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


enc = tok(texts, truncation=True, padding="max_length", max_length=32)
enc["labels"] = enc["input_ids"].copy()


train_ds = Dataset.from_dict(enc)
train_ds.save_to_disk("./toy_dataset")
print("Toy dataset saved -> ./toy_dataset") 