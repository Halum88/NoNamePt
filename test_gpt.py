from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt_text):
    # Загрузка токенизатора GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Загрузка модели GPT-2
    model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

    # Генерация текста
    output_text = model.generate(input_ids=tokenizer.encode(prompt_text, return_tensors="pt"), 
                                  max_length=100, 
                                  num_return_sequences=1, 
                                  temperature=1.0, 
                                  top_k=50, 
                                  top_p=0.95, 
                                  repetition_penalty=1.0, 
                                  num_beams=1)[0]

    # Декодирование и вывод сгенерированного текста
    generated_text = tokenizer.decode(output_text, skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    # Пример использования функции для генерации текста
    prompt_text = "Давным-давно, в далекой-далекой стране"
    generated_text = generate_text(prompt_text)
    print(generated_text)
