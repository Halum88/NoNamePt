import os
import telebot
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dotenv import load_dotenv

load_dotenv()

bot_API = os.getenv("TELEGRAM_BOT_TOKEN")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

bot_token = bot_API
bot = telebot.TeleBot(bot_token)

def generate_response(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text



@bot.message_handler(func=lambda message: True)
def greet_user(message):
    try:
        response = generate_response(message.text)
        bot.send_message(message.chat.id, response)
    except Exception as e:
        print(f"An error occurred greet_user: {e}")


# run bot
try:
    bot.polling(none_stop=True)
except Exception as e:
    print(f"An error occurred while polling: {e}")
