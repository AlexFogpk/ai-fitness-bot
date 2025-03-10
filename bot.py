from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
import logging
import asyncio
import os
import firebase_admin
from firebase_admin import credentials, firestore

# Инициализация Firebase
cred = credentials.Certificate('firebase.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Получаем токен из переменных окружения Railway
TOKEN = os.getenv("BOT_TOKEN")
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Логирование
logging.basicConfig(level=logging.INFO)

# Реакция на команду /start
@dp.message(CommandStart())
async def start(message: types.Message):
    db = firestore.client()
    user_id = str(message.from_user.id)

    user_ref = db.collection('users').document(user_id)
    user_ref.set({
        "name": message.from_user.full_name,
        "telegram_id": user_id,
        "subscription": "free",
    }, merge=True)

    await message.answer(f"Привет, {message.from_user.full_name}! 👋 Я AI-бот по похудению. Скоро тут появятся персональные советы!")

# Простой тестовый ответ
@dp.message()
async def default_answer(message: types.Message):
    await message.answer("Это твой AI-бот по похудению! Скоро здесь появится умный помощник.")

async def main():
    logging.basicConfig(level=logging.INFO)
    BOT_TOKEN = os.getenv("BOT_TOKEN")
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()
    
    # Инициализация Firebase
    firebase_credentials = os.getenv("FIREBASE_CREDENTIALS")
    with open("firebase.json", "w") as f:
        f = f"{firebase_credentials}"
        f = open("firebase.json", "w")
        f.write(os.getenv("FIREBASE_CREDENTIALS"))
        f.close()

    cred = credentials.Certificate("firebase.json")
    firebase_admin.initialize_app(cred)

    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
