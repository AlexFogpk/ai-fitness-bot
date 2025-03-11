from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from aiogram.enums import ParseMode  # Добавляем импорт ParseMode
import logging
import asyncio
import os
import re
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

# Инициализация OpenAI клиента (GPT-4o-mini)
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Инициализация Telegram-бота
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
logging.basicConfig(level=logging.INFO)

# ----------------------------------------------------------
# 1. Функция разбивки длинного текста на части (для Telegram лимита 4096 символов)
# ----------------------------------------------------------
def split_message(text, max_length=4096):
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
# 2. Функция отправки сообщений по частям с поддержкой Markdown
# ----------------------------------------------------------
async def send_split_message(chat_id, text, parse_mode=None):
    parts = split_message(text, max_length=4096)
    for part in parts:
        await bot.send_message(chat_id, part, parse_mode=parse_mode)

# ----------------------------------------------------------
# 3. Функция пост-обработки Markdown для оптимизации шрифта в Telegram
# ----------------------------------------------------------
def fix_markdown_telegram(text: str) -> str:
    lines = text.split("\n")
    new_lines = []
    for line in lines:
        if line.startswith("### "):
            heading = line[4:].strip()
            line = f"**{heading}**"
        elif line.startswith("## "):
            heading = line[3:].strip()
            line = f"**{heading}**"
        new_lines.append(line)
    return "\n".join(new_lines)

# ----------------------------------------------------------
# 4. Функция проверки тематичности через регулярные выражения
# ----------------------------------------------------------
def is_topic_by_regex(text: str) -> bool:
    patterns = [
        # Фитнес, тренировки, физкультура
        r"\bфитнес\w*",
        r"\bтрениров\w*",
        r"\bтренир\w*",
        r"\bупражн\w*",
        r"\bфизкульт\w*",
        r"\bспорт\w*",
        # Диета, питание, здоровье
        r"\bдиет\w*",
        r"\bпитан\w*",
        r"\bкалор\w*",
        r"\bбелк\w*",
        r"\bжир\w*",
        r"\bуглевод\w*",
        r"\bпротеин\w*",
        r"\bпохуд\w*",
        r"\bздоров\w*",
        r"\bмышц\w*",
        r"\bмасс\w*",
        r"\bрацион\w*",
        r"\bвитамин\w*",
        r"\bминерал\w*",
        r"\bгидрат\w*",
        r"\bсон\w*",
        r"\bотдых\w*",
        r"\bвосстановлен\w*",
        # Диетические режимы
        r"\bвеган\w*",
        r"\bвегетариан\w*",
        r"\bкето\w*",
        # Кардио и силовые тренировки
        r"\bкардио\w*",
        r"\bhiit\b",
        r"\bсил\w*",
        r"\bпресс\w*",
        r"\bягодиц\w*",
        r"\bрастяжк\w*",
        r"\bфункционал\w*",
        # Спортивное питание и добавки
        r"\bсуплемент\w*",
        r"\bдобавк\w*",
        r"\bанабол\w*",
        r"\bжиросжиган\w*",
        # Дополнительные аспекты тренировок
        r"\bгипертроф\w*",
        r"\bвыносливост\w*",
        r"\bплиометрик\w*",
        r"\bдинамик\w*",
        r"\bскорост\w*",
        # Спортивные дисциплины и соревнования
        r"\bфутбол\w*",
        r"\bбокс\w*",
        r"\bбег\w*",
        r"\bвелосипед\w*",
        r"\bплавани\w*",
        r"\bлыжи\w*",
        r"\bсоревнован\w*",
        r"\батлет\w*",
        r"\bспортсмен\w*",
        r"\bэкипировк\w*",
        r"\bинвентарь\w*",
        r"\bсостязани\w*",
        r"\bтурнир\w*",
        r"\bфизическ\w*"
    ]
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in patterns)

# ----------------------------------------------------------
# 5. Функция проверки тематичности через GPT (fallback)
# ----------------------------------------------------------
async def is_topic_by_gpt(text: str) -> bool:
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Ты эксперт по фитнесу, тренировкам, здоровью и питанию. Отвечай только 'да' или 'нет'."
            },
            {
                "role": "user",
                "content": f"Относится ли следующий текст к теме фитнеса, тренировкам, здоровью или питанию? Текст: {text}"
            }
        ],
        temperature=0,
        max_tokens=10
    )
    answer = response.choices[0].message.content.strip().lower()
    return "да" in answer

# ----------------------------------------------------------
# 6. Комбинированная функция проверки тематичности
# ----------------------------------------------------------
async def is_fitness_question_combined(text: str) -> bool:
    if is_topic_by_regex(text):
        return True
    else:
        return await is_topic_by_gpt(text)

# ----------------------------------------------------------
# 7. Функция для получения ответа от OpenAI (GPT-4o-mini)
# ----------------------------------------------------------
async def ask_gpt(user_message: str) -> str:
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты профессиональный AI-тренер, консультируешь по фитнесу, тренировкам, здоровью и питанию. "
                    "Отвечай кратко и структурированно, используя Markdown, совместимый с Telegram. "
                    "Не используй заголовки вида '###'. Вместо этого для заголовков используй жирный текст. "
                    "Если вопрос не по теме, отвечай: 'Извини, я могу отвечать только на вопросы о фитнесе, тренировках и здоровом образе жизни.'"
                )
            },
            {"role": "user", "content": user_message}
        ],
        temperature=0.5,
        max_tokens=1000
    )
    return response.choices[0].message.content

# ----------------------------------------------------------
# 8. Стартовая команда
# ----------------------------------------------------------
@dp.message(CommandStart())
async def start(message: types.Message):
    user_id = str(message.from_user.id)
    user_ref = db.collection('users').document(user_id)
    user_ref.set({
        "name": message.from_user.full_name,
        "telegram_id": user_id,
        "subscription": "free",
    }, merge=True)
    await message.answer(
        f"Привет, {message.from_user.full_name}! 👋\n\n"
        "Я твой AI-тренер. Спроси меня о *питании*, _тренировках_, похудении, здоровье или спорте!",
        parse_mode=ParseMode.MARKDOWN
    )

# ----------------------------------------------------------
# 9. Основной обработчик сообщений с комбинированной фильтрацией тематики
# ----------------------------------------------------------
@dp.message()
async def handle_message(message: types.Message):
    if not await is_fitness_question_combined(message.text):
        await message.answer(
            "Извини, я могу отвечать только на вопросы, связанные с фитнесом, тренировками, диетой, похудением, здоровьем и спортом.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    await message.chat.do("typing")
    response = await ask_gpt(message.text)
    clean_response = fix_markdown_telegram(response)
    await send_split_message(message.chat.id, clean_response, parse_mode=ParseMode.MARKDOWN)

# ----------------------------------------------------------
# 10. Запуск бота
# ----------------------------------------------------------
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
