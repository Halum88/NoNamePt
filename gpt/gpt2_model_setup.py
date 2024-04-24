from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Loading GPT-2 tokeniser
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Loading GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
