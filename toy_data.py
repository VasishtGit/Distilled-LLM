from datasets import Dataset
from transformers import AutoTokenizer

# Example training sentences
sentences = [
    "AI will change the future of humanity.",
    "Knowledge distillation makes small models smarter.",
    "Kalman filters smooth noisy signals.",
    "Transformers are powerful sequence models.",
    "Small models are faster and lighter to run.",
] 

# Load GPT-2's tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Some tokenizers like GPT 2 don’t have a [PAD] token by default
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # reuse end-of-sequence as padding

# Step 1: Turn text into numbers
tokenized_batch = tokenizer(
    sentences,
    truncation=True,            # cut off if sentence is too long
    padding="max_length",       # pad shorter ones to the same length
    max_length=32               # everything becomes length 32
)

# Step 2: For language modeling, labels = inputs (predict the next word)
tokenized_batch["labels"] = tokenized_batch["input_ids"].copy()

# Step 3: Convert dictionary into a Dataset object
training_dataset = Dataset.from_dict(tokenized_batch)

# Step 4: Save dataset to disk
training_dataset.save_to_disk("./toy_dataset")
print("Toy dataset saved -> ./toy_dataset")
