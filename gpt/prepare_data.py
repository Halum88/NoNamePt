import torch
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# load data
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=512):
        self.tokenizer = tokenizer
        self.block_size = block_size

        # read data is file
        with open(file_path, "r", encoding="utf-8") as file:
            self.examples = file.readlines()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx].strip()

        inputs = self.tokenizer(example, return_tensors="pt", padding=True, truncation=True, max_length=self.block_size)
        return inputs


today = datetime.now().strftime("%Y-%m-%d")
file_path = f"./source/tokenized_texts_{today}.txt"

# create Dataset
dataset = TextDataset(file_path, tokenizer)

# dynamic packaging
def dynamic_collate_fn(batch):
    max_len = max(len(inputs["input_ids"][0]) for inputs in batch)
    padded_inputs = {
        key: torch.stack([torch.cat([inputs[key][0], torch.zeros(max_len - len(inputs[key][0]), dtype=torch.long)]) for inputs in batch])
        for key in batch[0].keys()
    }
    return padded_inputs

# create DataLoader for learn dynamic packaging
train_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dynamic_collate_fn)

# example for DataLoader
for batch in train_loader:
    print(batch)
