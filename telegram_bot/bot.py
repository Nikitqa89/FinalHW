import os
import asyncio
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import CommandStart, Command
from transformer_model.generate import generate_text

# Загрузка токена из .env
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("❌ TELEGRAM_BOT_TOKEN не найден в .env")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# 🧠 Параметры генерации по умолчанию
user_settings = {}

DEFAULT_PARAMS = {
    "temperature": 1.0,
    "top_k": 40,
    "max_new_tokens": 200,
}

# =====================
# Команды Telegram
# =====================
async def set_commands(bot: Bot):
    commands = [
        types.BotCommand(command="start", description="Начать общение с ботом"),
        types.BotCommand(command="set", description="Установить параметры генерации"),
        types.BotCommand(command="params", description="Показать текущие параметры"),
        types.BotCommand(command="stanza", description="Сгенерировать одну строфу"),
        types.BotCommand(command="help", description="Показать список команд"),
    ]
    await bot.set_my_commands(commands)

# =====================
# /start
# =====================
@dp.message(CommandStart())
async def start_handler(message: Message):
    user_id = message.from_user.id
    user_settings[user_id] = DEFAULT_PARAMS.copy()
    await message.answer("Привет! Напиши строку, и я продолжу её стихами ✨\n\nЧтобы увидеть команды — отправь /help")

# =====================
# /help
# =====================
@dp.message(Command("help"))
async def help_handler(message: Message):
    await message.answer(
        "📜 Доступные команды:\n"
        "/start — начать общение\n"
        "/set temperature=0.8 top_k=50 — задать параметры генерации\n"
        "/params — посмотреть текущие параметры\n"
        "/stanza — сгенерировать одну поэтическую строфу\n"
        "/help — показать это меню"
    )

# =====================
# /set
# =====================
@dp.message(Command("set"))
async def set_params(message: Message):
    user_id = message.from_user.id
    text = message.text.lower().replace("/set", "").strip()

    if user_id not in user_settings:
        user_settings[user_id] = DEFAULT_PARAMS.copy()

    parts = text.split()
    updates = []
    for part in parts:
        if "=" in part:
            key, val = part.split("=", 1)
            if key in DEFAULT_PARAMS:
                try:
                    user_settings[user_id][key] = float(val)
                    updates.append(f"{key} = {val}")
                except ValueError:
                    pass

    if updates:
        await message.answer("🔧 Обновлено:\n" + "\n".join(updates))
    else:
        await message.answer("⚠️ Используй формат: /set temperature=0.8 top_k=50")

# =====================
# /params
# =====================
@dp.message(Command("params"))
async def show_params(message: Message):
    user_id = message.from_user.id
    settings = user_settings.get(user_id, DEFAULT_PARAMS)
    text = "\n".join([f"{k}: {v}" for k, v in settings.items()])
    await message.answer(f"⚙️ Текущие параметры:\n{text}")

# =====================
# /stanza — генерация до <EOS>
# =====================
@dp.message(Command("stanza"))
async def stanza_handler(message: Message):
    user_id = message.from_user.id
    prompt = message.text.replace("/stanza", "").strip()
    params = user_settings.get(user_id, DEFAULT_PARAMS)

    try:
        text = generate_text(
            prompt=prompt,
            max_new_tokens=300,  # с запасом, генерация остановится по <EOS>
            temperature=params["temperature"],
            top_k=int(params["top_k"])
        )
        await message.answer(text)
    except Exception as e:
        print(f"[Ошибка генерации строфы]: {e}")
        await message.answer("Что-то пошло не так 🤖")

# =====================
# Генерация на любое сообщение
# =====================
@dp.message()
async def text_handler(message: Message):
    user_id = message.from_user.id
    prompt = message.text.strip()

    if not prompt:
        await message.answer("Пожалуйста, напиши хоть что-нибудь")
        return

    params = user_settings.get(user_id, DEFAULT_PARAMS)

    try:
        generated = generate_text(
            prompt=prompt,
            max_new_tokens=int(params["max_new_tokens"]),
            temperature=params["temperature"],
            top_k=int(params["top_k"])
        )
        await message.answer(generated)

    except ValueError as ve:
        await message.answer(str(ve))

    except Exception as e:
        print(f"[Ошибка генерации]: {e}")
        await message.answer("Ой, что-то пошло не так. Попробуй снова!")

# =====================
# Запуск бота
# =====================
async def main():
    print("🚀 Бот запущен!")
    await set_commands(bot)
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Бот остановлен вручную.")