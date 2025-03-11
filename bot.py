from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
import logging
import asyncio
import os
import firebase_admin
from firebase_admin import credentials, firestore
from openai import AsyncOpenAI

# Переменные окружения
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Firebase инициализация
firebase_credentials = os.getenv("FIREBASE_CREDENTIALS")
with open("firebase.json", "w") as f:
    f.write(firebase_credentials)

cred = credentials.Certificate('firebase.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Инициализация GPT-4o mini
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Инициализация Telegram-бота
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

logging.basicConfig(level=logging.INFO)

# ----------------------------------------------------------
# 1. Функция разбивки длинного текста на части
# ----------------------------------------------------------
def split_message(text, max_length=4096):
    """
    Разбивает длинный текст на части, каждая из которых не превышает max_length символов.
    Если возможно, разбивает по переносам строки, чтобы не рвать слово.
    """
    parts = []
    while len(text) > max_length:
        split_index = text.rfind("\n", 0, max_length)
        if split_index == -1:
            split_index = max_length
        parts.append(text[:split_index])
        text = text[split_index:].strip()
    parts.append(text)
    return parts

# ----------------------------------------------------------
# 2. Функция отправки сообщений по частям
# ----------------------------------------------------------
async def send_split_message(chat_id, text):
    """
    Отправляет длинное сообщение в чат, разбивая его на части,
    чтобы каждая часть не превышала 4096 символов.
    """
    parts = split_message(text, max_length=4096)
    for part in parts:
        await bot.send_message(chat_id, part)

# Ответ GPT-4o mini
async def ask_gpt(user_message: str) -> str:
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ты профессиональный AI-тренер, консультируешь по похудению, фитнесу и питанию."},
            {"role": "user", "content": user_message}
        ],
        temperature=0.5,
        max_tokens=500
    )
    return response.choices[0].message.content

# Стартовая команда
@dp.message(CommandStart())
async def start(message: types.Message):
    user_id = str(message.from_user.id)

    user_ref = db.collection('users').document(user_id)
    user_ref.set({
        "name": message.from_user.full_name,
        "telegram_id": user_id,
        "subscription": "free",
    }, merge=True)

    await message.answer(f"Привет, {message.from_user.full_name}! 👋 Я твой AI-тренер. Спроси меня о питании, тренировках или похудении!")

# Основной обработчик сообщений (ответ через GPT-4o mini)
@dp.message()
async def handle_message(message: types.Message):
    await message.chat.do("typing")
    response = await ask_gpt(message.text)
    
    # ----------------------------------------------------------
    # 3. Вместо message.answer(response) отправляем сообщениями по частям
    # ----------------------------------------------------------
    await send_split_message(message.chat.id, response)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
