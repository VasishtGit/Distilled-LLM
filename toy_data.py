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

#GPT-2 has no pad token, so we borrow EOS
#EOS or Padding is used as GPT 2 was trained using continuous text and not fixed length texts.
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

#Tokenize
enc = tok(texts, truncation=True, padding="max_length", max_length=32)

# Copy input_ids into labels (for next-token prediction)
enc["labels"] = enc["input_ids"].copy()

# Build dataset
train_ds = Dataset.from_dict(enc)

#Print one example
print("One example row:")
print(train_ds[0])  

#Decode back to text so it makes sense
print("\nDecoded back to text:")
print(tok.decode(train_ds[0]["input_ids"]))

#Saving dataset to disk for future training
train_ds.save_to_disk("./toy_dataset")
print("\nToy dataset saved -> ./toy_dataset")
