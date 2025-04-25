# 🧠 GPT-Бот для генерации стихов

Нейросеть, обученная писать поэзию на русском языке.  
Бот использует GPT-like трансформер и Telegram-интерфейс на `aiogram`.

---

## 📁 Структура проекта

```
.
├── bot.py                    # Telegram-бот (aiogram 3)
├── config.py                 # Гиперпараметры модели
├── data.txt                  # Сырой корпус
├── filtered_poetry_lines.txt # Очищенные строки
├── generate.py               # Генерация текста
├── model.py                  # Архитектура GPT
├── preprocess.py             # Очистка и фильтрация текста
├── tokenizer.py              # Словарь и токенизация
├── train.py                  # Обучение модели
├── models/                   # Каталог с весами модели
└── .env                      # Переменные окружения (Telegram токен)
```

---

## 🔧 Установка

```bash
pip install -r requirements.txt
```

Создайте `.env`:

```env
TELEGRAM_BOT_TOKEN=ваш_токен_бота
```

---

## ⚙️ Предобработка данных

```bash
python preprocess.py
```

Файл `filtered_poetry_lines.txt` будет создан с отфильтрованными строками на основе правил (длина, символы, отсутствие латиницы и пр.).

---

## 🧠 Обучение модели

```bash
python train.py
```

Сохраняются веса:  
- `models/model.pth` — последняя  
- `models/best_model.pth` — с лучшим `val loss`  
- `loss_plot.png` — график потерь

---

## ✨ Генерация

```python
from generate import generate_text, generate_batch

generate_text(
    prompt="Ты идешь",
    temperature=0.9,
    top_k=10,
    max_new_tokens=100,
    eos_token="\n"
)

# Пакетная генерация
generate_batch([
    "Ты идешь по дороге",
    "Небо роняет слезы",
    "Осень в сердце"
], temperature=0.9, top_k=10)
```

---

## 🤖 Telegram-бот

Запуск:

```bash
python bot.py
```

---

## 📲 Команды в Telegram

| Команда      | Описание                                                       |
|--------------|----------------------------------------------------------------|
| `/start`     | Начать общение с ботом                                         |
| `/help`      | Показать справку                                               |
| `/set`       | Изменить параметры (`temperature`, `top_k`, `max_new_tokens`) |
| `/params`    | Показать текущие параметры                                     |
| `/stanza`    | Сгенерировать одну строфу (до символа новой строки)           |
| `/batch`     | Пакетная генерация: каждая строка — отдельный prompt          |

---

## 📒 Примеры использования в боте

- `/set temperature=1.0 top_k=15`
- `/batch\nТы идешь\nЛистья падают\nОсень рядом`

---

## 📌 Особенности

- Используется упрощённый словарь (~70 символов)
- Автообрезка длинного контекста (`block_size = 256`)
- `eos_token='\n'` завершает генерацию по строфе
- Печать в реальном времени: через `on_token`
- Логи сохраняются в `generation.log`

---

## 🛠 Требования

- Python 3.10+
- PyTorch
- aiogram 3.x

---
