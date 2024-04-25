import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

bot_dir = os.path.join(parent_dir, 'bot')
sys.path.append(bot_dir)


from database import create_connection
from transformers import GPT2Tokenizer


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def extract_messages():
    # connection to the DB
    connection = create_connection()
    if connection is None:
        print("Failed to connect to the db")
        return []

    messages = []
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT text FROM messages")
        messages = cursor.fetchall()
    except Exception as error:
        print("Error fetching messages from database:", error)
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
    return messages



def process_messages(messages):
    for text in messages:
        input_ids = tokenizer.encode(text[0], return_tensors="pt")
        print(text)
        print("Input IDs:", input_ids)



if __name__ == "__main__":
    messages = extract_messages()
    process_messages(messages)
