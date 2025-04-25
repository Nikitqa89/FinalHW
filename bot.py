import os
import asyncio
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import CommandStart, Command
from generate import generate_text, generate_batch  # ✅ импортируем пакетную генерацию

# =====================
# 🔐 Загрузка токена
# =====================
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("❌ TELEGRAM_BOT_TOKEN не найден в .env")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# =====================
# ⚙️ Параметры по умолчанию
# =====================
user_settings = {}

DEFAULT_PARAMS = {
    "temperature": 0.8,
    "top_k": 10,
    "max_new_tokens": 200,
}

# =====================
# 📋 Команды бота
# =====================
async def set_commands(bot: Bot):
    commands = [
        types.BotCommand(command="start", description="Начать общение с ботом"),
        types.BotCommand(command="set", description="Установить параметры генерации"),
        types.BotCommand(command="params", description="Показать текущие параметры"),
        types.BotCommand(command="stanza", description="Сгенерировать одну строфу"),
        types.BotCommand(command="batch", description="Пакетная генерация нескольких строк"),
        types.BotCommand(command="help", description="Показать список команд"),
    ]
    await bot.set_my_commands(commands)

# =====================
# /start — инициализация
# =====================
@dp.message(CommandStart())
async def start_handler(message: Message):
    user_id = message.from_user.id
    user_settings[user_id] = DEFAULT_PARAMS.copy()
    await message.answer("Привет! Напиши строку, и я продолжу её стихами ✨\n\nЧтобы увидеть команды — отправь /help")

# =====================
# /help — справка
# =====================
@dp.message(Command("help"))
async def help_handler(message: Message):
    await message.answer(
        "📜 Доступные команды:\n"
        "/start — начать общение\n"
        "/set temperature=0.8 top_k=10 — задать параметры генерации\n"
        "/params — посмотреть текущие параметры\n"
        "/stanza — сгенерировать одну поэтическую строфу\n"
        "/batch — сгенерировать сразу несколько строк (каждая с новой строки)\n"
        "/help — показать это меню"
    )

# =====================
# /set — настройка параметров
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
        await message.answer("⚠️ Используй формат: /set temperature=0.8 top_k=10")

# =====================
# /params — текущие настройки
# =====================
@dp.message(Command("params"))
async def show_params(message: Message):
    user_id = message.from_user.id
    settings = user_settings.get(user_id, DEFAULT_PARAMS)
    text = "\n".join([f"{k}: {v}" for k, v in settings.items()])
    await message.answer(f"⚙️ Текущие параметры:\n{text}")

# =====================
# /stanza — генерация до \n (EOS)
# =====================
@dp.message(Command("stanza"))
async def stanza_handler(message: Message):
    user_id = message.from_user.id
    prompt = message.text.replace("/stanza", "").strip()
    params = user_settings.get(user_id, DEFAULT_PARAMS)

    try:
        text = generate_text(
            prompt=prompt,
            max_new_tokens=300,
            temperature=params["temperature"],
            top_k=int(params["top_k"]),
            eos_token='\n'
        )
        await message.answer(text)
    except Exception as e:
        print(f"[Ошибка генерации строфы]: {e}")
        await message.answer("Что-то пошло не так 🤖")

# =====================
# /batch — пакетная генерация
# =====================
@dp.message(Command("batch"))
async def batch_handler(message: Message):
    user_id = message.from_user.id
    prompts = message.text.replace("/batch", "").strip().split('\n')
    prompts = [p.strip() for p in prompts if p.strip()]

    if not prompts:
        await message.answer("⚠️ Введите строки после /batch, каждую с новой строки.")
        return

    params = user_settings.get(user_id, DEFAULT_PARAMS)

    try:
        results = generate_batch(
            prompts,
            max_new_tokens=int(params["max_new_tokens"]),
            temperature=params["temperature"],
            top_k=int(params["top_k"]),
            eos_token='\n'
        )
        reply = ""
        for i, (prompt, gen) in enumerate(zip(prompts, results), 1):
            reply += f"*{i}. {prompt}*\n{gen.strip()}\n\n"
        await message.answer(reply.strip(), parse_mode="Markdown")
    except Exception as e:
        print(f"[Ошибка batch генерации]: {e}")
        await message.answer("⚠️ Что-то пошло не так при пакетной генерации.")

# =====================
# 🔤 Генерация на любое сообщение
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
            top_k=int(params["top_k"]),
            eos_token='\n'
        )
        await message.answer(generated)

    except ValueError as ve:
        await message.answer(str(ve))

    except Exception as e:
        print(f"[Ошибка генерации]: {e}")
        await message.answer("Ой, что-то пошло не так. Попробуй снова!")

# =====================
# 🚀 Запуск бота
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