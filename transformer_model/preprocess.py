"""
Модуль предобработки текста для обучения языковой модели.

Функции:
- Очистка поэтического текста от мусора
- Удаление заголовков (опционально)
- Построение словаря символов (vocab)
- Кодирование текста в тензоры PyTorch
- Сохранение готовых данных для обучения
"""

import re
import torch
from config import *
import numpy as np

# ============================================
# 🧹 Расширенная очистка поэтического текста
# ============================================

def preprocess_poetry(text: str) -> str:
    """
    Очищает текст:
    - удаляет ссылки, рекламу, сноски, латиницу и цифры
    - убирает редакторские вставки и нестрофические вставки
    - фильтрует заголовки, если включено
    - форматирует по строфам (по двойным переносам)
    """
    # Удаляем URL и рекламу
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'(скачали.*?|все книги автора.*?|приятного чтения.*?|благодарим вас.*?)\n', '', text, flags=re.IGNORECASE)

    # Удаляем сноски и комментарии
    text = re.sub(r'\[\d+.*?\]', '', text)
    text = re.sub(r'\[.*?\]', '', text)  # произвольные скобки []

    # Удаляем редакторские вставки и "От редакции"
    text = re.sub(r'От редакции.*?(\n{2,}|\Z)', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Удаляем латиницу, цифры, и лишние символы
    text = re.sub(r'[A-Za-z0-9]', '', text)
    text = re.sub(r'[^\nА-яЁё—.,:;!?()"\- ]+', '', text)

    # Удаляем 3+ пустых строк, оставляем максимум двойной перенос
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ ]{2,}', ' ', text)

    # Удаляем заголовки В ВЕРХНЕМ РЕГИСТРЕ до 7 слов
    if remove_titles:
        lines = text.splitlines()
        text = '\n'.join([
            line for line in lines
            if not (
                line.isupper()
                and len(line.strip().split()) <= 7
                and not any(p in line for p in '.,:;!?()')
            )
        ])

    # Удаляем строки с очень длинными словами (мусор)
    text = '\n'.join([
        line for line in text.splitlines()
        if line.strip() and all(len(word) < 25 for word in line.split())
    ])

    # Убираем одиночные строки/буквы
    text = '\n'.join([line for line in text.splitlines() if len(line.strip()) > 1])

    # Формируем строфы по двойным переносам
    stanzas = text.split('\n\n')
    clean_stanzas = [s.strip() for s in stanzas if len(s.strip().splitlines()) >= 2]
    return '<EOS>'.join(clean_stanzas).strip()

# ============================
# 🔠 Построение словаря (vocab)
# ============================

def build_vocab(text):
    if "<EOS>" not in text:
        text += "<EOS>"  # гарантируем, что войдёт в словарь
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

# ===========================
# 🔁 Кодирование и декодирование
# ===========================

def encode(text, stoi):
    return [stoi[c] for c in text]

def decode(indices, itos):
    return ''.join([itos[i] for i in indices])

# ========================
# 📦 Подготовка датасета
# ========================

def prepare_data(text, stoi):
    data = torch.tensor(encode(text, stoi), dtype=torch.long)

    # Индексы возможных начальных позиций для фрагментов длиной block_size
    indices = np.arange(len(data) - block_size)
    np.random.seed(42)
    np.random.shuffle(indices)

    # Разбиваем индексы 80/20
    split_idx = int(0.8 * len(indices))
    train_ix = indices[:split_idx]
    val_ix = indices[split_idx:]

    # Собираем тензоры
    train_data = torch.cat([data[i:i + block_size] for i in train_ix])
    val_data = torch.cat([data[i:i + block_size] for i in val_ix])

    return train_data, val_data

# ====================
# 🚀 Запуск скрипта
# ====================

if __name__ == "__main__":
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    clean_text = preprocess_poetry(raw_text)
    stoi, itos = build_vocab(clean_text)
    train_data, val_data = prepare_data(clean_text, stoi)

    torch.save((train_data, val_data, stoi, itos), 'data_prepared.pt')
    print("✅ Данные предобработаны и сохранены.")