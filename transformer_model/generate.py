"""
Модуль генерации текста на основе обученной GPT-языковой модели.

Функция generate_text(prompt) используется для генерации продолжения
введённого текста с помощью вероятностного сэмплирования.
"""

import os
import torch
import re
from transformer_model.model import GPTLanguageModel
from config import *

# ============================
# 📥 Загрузка обученных данных
# ============================

base_dir = os.path.dirname(__file__)
data_prepared_path = os.path.join(base_dir, "data_prepared.pt")
model_checkpoint_path = os.path.join(base_dir, "best_model.pt" if load_best_model else "model.pt")

train_data, val_data, stoi, itos = torch.load(data_prepared_path)
vocab_size = len(stoi)

model = GPTLanguageModel(vocab_size)
model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
model.eval().to(device)

# ========================
# 🔠 Кодировщики текста
# ========================

def encode(s: str) -> list[int]:
    encoded = [stoi[c] for c in s if c in stoi]
    if not encoded:
        raise ValueError("⛔ В тексте нет допустимых символов для генерации.")
    return encoded

def decode(indices: list[int]) -> str:
    return ''.join([itos[i] for i in indices])

# ========================
# 🧠 Генерация текста моделью
# ========================

@torch.no_grad()
def generate_text(prompt: str = "", max_new_tokens: int = 100, temperature: float = 1.0, top_k: int = 10) -> str:
    """
    Генерирует продолжение текста с помощью модели.
    """
    idx = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    eos_token = stoi.get('<EOS>')

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)

        if top_k is not None:
            top_probs, top_idx = torch.topk(probs, k=top_k, dim=-1)
            top_probs = top_probs / torch.sum(top_probs, dim=-1, keepdim=True)
            next_token = top_idx.gather(-1, torch.multinomial(top_probs, num_samples=1))
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        if eos_token is not None and next_token.item() == eos_token:
            break

        idx = torch.cat((idx, next_token), dim=1)

    raw_text = decode(idx[0].tolist())

    # Заменяем <EOS> на перенос строки
    final_text = raw_text.replace('<EOS>', '\n\n')

    # Чистим лишние пробелы и переносы
    lines = [line.strip() for line in final_text.split('\n')]
    formatted_lines = [line for line in lines if len(line) > 1]
    clean_text = '\n'.join(formatted_lines)
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text).strip()

    return f"{clean_text}"