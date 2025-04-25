import torch
import logging
from model import GPTLanguageModel
from config import *
from tokenizer import build_vocab, encode, decode

# =============================
# 📦 Логирование
# =============================
logging.basicConfig(level=logging.INFO, filename="generation.log", filemode="a", format="%(asctime)s [%(levelname)s] %(message)s")

# =============================
# 📦 Загрузка модели и словаря
# =============================

with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

stoi, itos = build_vocab(text)
vocab_size = len(stoi)

# =======================
# 🧐 Загрузка модели
# =======================

def load_model():
    model = GPTLanguageModel(vocab_size)
    path = "models/best_model.pth" if load_best_model else "models/model.pth"
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    logging.info(f"✅ Модель загружена с {path}")
    return model

# =======================
# ✨ Генерация текста
# =======================

def generate_text(prompt: str = '', max_new_tokens=200, temperature=1.0, top_k=10, on_token=None) -> str:
    if not prompt.strip():
        return "⚠️ Пожалуйста, введите начальный текст"

    if not (0.1 <= temperature <= 5.0):
        return "⛔️ Температура должна быть от 0.1 до 5.0"
    if not (1 <= top_k <= vocab_size):
        return f"⛔️ top_k должен быть от 1 до {vocab_size}"

    model = load_model()
    encoded_prompt = encode(prompt, stoi)

    unknown_count = len(prompt) - len(encoded_prompt)
    if unknown_count > 0:
        logging.warning(f"⚠️ {unknown_count} неизвестных символов будут проигнорированы")

    context = torch.tensor([encoded_prompt], dtype=torch.long).to(device)

    if context.size(1) > block_size:
        logging.warning(f"⚠️ Контекст усекается до {block_size} токенов")
        context = context[:, -block_size:]

    logging.info(f"🔮 Начинается генерация: prompt='{prompt}'")
    generated = context

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = generated[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(1, indices, values)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

            next_char = itos.get(next_token.item(), '')
            if on_token:
                on_token(next_char)
            else:
                print(next_char, end='', flush=True)

            if next_char == '\n':
                break

    logging.info(f"✅ Генерация завершена")
    decoded = decode(generated[0].tolist(), itos).strip()
    if not decoded:
        return "⚠️ Ничего не сгенерировалось."
    return decoded

# =======================
# 🥳 Пакетная генерация
# =======================
def generate_batch(prompts: list[str], **params) -> list[str]:
    return [generate_text(prompt=p, **params) for p in prompts]
