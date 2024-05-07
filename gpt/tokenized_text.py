import sys
import os
from datetime import datetime


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

bot_dir = os.path.join(parent_dir, 'bot')
sys.path.append(bot_dir)


from database import create_connection # type: ignore
from transformers import GPT2Tokenizer


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def extract_messages():
    # connection to the DB
    connection = create_connection()
    today = datetime.now().strftime("%Y-%m-%d")
    if connection is None:
        print("Failed to connect to the db")
        return []

    messages = []
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT text FROM messages WHERE created_at::date = %s", (today,))
        messages = cursor.fetchall()
    except Exception as error:
        print("Error fetching messages from database:", error)
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
    return messages


# ref token text
def process_messages(messages):
    tokenized_texts = []
    for text in messages:
        input_ids = tokenizer.encode(text[0], return_tensors="pt")
        tokenized_texts.append(input_ids)
    return tokenized_texts


def save_texts_to_file(texts, directory):
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"tokenized_texts_{today}.txt"

    filepath = os.path.join(directory, filename)

    with open(filepath, mode='a', encoding='utf-8') as file:
        for text_tensor in texts:
            text_ids = text_tensor.flatten().tolist()
            text_str = ''
            for token_id in text_ids:
                text_str += str(token_id) + '; '
            text_str = text_str.rstrip('; ')
            file.write(text_str + '\n')



if __name__ == "__main__":
    messages = extract_messages()
    tokenized_texts = process_messages(messages)
    save_texts_to_file(tokenized_texts, './source')
