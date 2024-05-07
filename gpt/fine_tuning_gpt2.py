import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from torch.utils.data import DataLoader, RandomSampler
from datetime import datetime
import logging
from prepare_data import TextDataset, dynamic_collate_fn
import os

today = datetime.now().strftime("%Y-%m-%d")

logging.basicConfig(filename=f'./gpt/log/fine_tuning_{today}.log', level=logging.INFO)
output_model_path = f"./gpt/fine_tuned/fine_tuned_model.pt"


try:
    # Defining training parameters
    num_epochs = 3
    learning_rate = 1e-4
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading the tokeniser and model configuration
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    config = GPT2Config.from_pretrained("gpt2-medium")
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium", config=config)
    model.to(device)

    # loading data
    file_path = f"./source/tokenized_texts_{today}.txt" # file tokeniser file
    dataset = TextDataset(file_path, tokenizer)
    train_sampler = RandomSampler(dataset)
    train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size, collate_fn=dynamic_collate_fn)

    # Definition of loss function and optimiser
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Check if the model file exists
    if os.path.exists(output_model_path):
        model.load_state_dict(torch.load(output_model_path))
        logging.info("Pre-trained model loaded from existing file.")
    else:
        logging.info("Pre-trained model file not found. Training from scratch.")

    # init training process
    logging.info(f"Starting or continuing fine-tuning_{today}")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        try:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = input_ids.clone().detach()
                labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding tokens for loss calculation
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
        except Exception as e:
            print(f'Error proccess learn ----  {e}')

        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}")

    # Saving the pre-trained model
    torch.save(model.state_dict(), output_model_path)
    logging.info(f"Fine-tuned model saved to {output_model_path}")

except Exception as e:
    logging.error(f"An error occurred during fine-tuning: {e}")
