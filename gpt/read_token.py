from datetime import datetime

today = datetime.now().strftime("%Y-%m-%d")
filename = f"tokenized_texts_{today}.txt"


def read_tokenized_texts(file_path):
    tokenized_texts = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            tokens = [int(token.strip()) for token in line.split(';')]
            tokenized_texts.append(tokens)
    return tokenized_texts

if __name__ == "__main__":
    file_path = f'./source/{filename}'
    tokenized_texts = read_tokenized_texts(file_path)
    print("read token:", tokenized_texts)
