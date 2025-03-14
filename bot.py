import logging
import asyncio
import os
import re
import difflib  # для fuzzy matching

from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart, Command
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State

import firebase_admin
from firebase_admin import credentials, firestore

# Для работы с OpenAI (gpt-4o-mini)
from openai import AsyncOpenAI

# Для работы с NLP (Hugging Face Transformers)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# =========================================
# Константы окружения
# =========================================
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =========================================
# Firebase инициализация
# =========================================
firebase_credentials = os.getenv("FIREBASE_CREDENTIALS")
if firebase_credentials:
    with open("firebase.json", "w") as f:
        f.write(firebase_credentials)
else:
    print("Внимание: переменная FIREBASE_CREDENTIALS не установлена!")

cred = credentials.Certificate("firebase.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# =========================================
# Инициализация OpenAI (GPT-4o-mini)
# =========================================
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# =========================================
# Инициализация NLP-модели (Hugging Face Transformers)
# =========================================
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

def nlp_is_fitness_topic(text: str) -> bool:
    """
    Используем модель для сентимент-анализа в качестве примера.
    Если модель возвращает "POSITIVE", считаем, что текст относится к фитнес-тематике.
    (Учтите, что для русского языка и точного определения 'фитнес/не фитнес'
    нужна дообученная модель или другая подходящая модель.)
    """
    result = classifier(text)
    label = result[0]["label"]
    return label == "POSITIVE"

# =========================================
# Глобальные переменные для приветствий (fuzzy matching)
# =========================================
GREETINGS = [
    "привет", "здравствуйте", "добрый день", "доброе утро", "хай", "приветствую",
    "здарова", "салют", "хелло", "хелоу", "хей", "хэй", "йоу",
    "hello", "hi", "hey", "good morning", "good day"
]

def is_greeting_fuzzy(text: str) -> bool:
    """
    Функция возвращает True, если текст (с учетом возможных опечаток)
    достаточно похож на одно из приветствий в GREETINGS.
    """
    text_lower = text.lower().strip()
    matches = difflib.get_close_matches(text_lower, GREETINGS, n=1, cutoff=0.8)
    return len(matches) > 0

# =========================================
# Инициализация бота + FSM
# =========================================
bot = Bot(token=BOT_TOKEN)
storage = MemoryStorage()  # Храним состояние в памяти
dp = Dispatcher(storage=storage)
logging.basicConfig(level=logging.INFO)

# =========================================
# 1. Функция разбивки длинного текста
# =========================================
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

# =========================================
# 2. Отправка сообщений по частям
# =========================================
async def send_split_message(chat_id, text, parse_mode=None):
    parts = split_message(text)
    for part in parts:
        await bot.send_message(chat_id, part, parse_mode=parse_mode)

# =========================================
# 3. Обработка Markdown (замена '###' -> '**')
# =========================================
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

# =========================================
# 4. Проверка тематичности (регулярки)
# =========================================
def is_topic_by_regex(text: str) -> bool:
    # Расширенный список, включающий фитнес, тренировки, здоровье и питание
    patterns = [
        # Фитнес, тренировки, здоровье
        r"\bфитнес\w*", r"\bтрениров\w*", r"\bтренир\w*", r"\bупражн\w*",
        r"\bфизкульт\w*", r"\bспорт\w*", r"\bдиет\w*", r"\bпитан\w*", r"\bкалор\w*",
        r"\bбелк\w*", r"\bжир\w*", r"\bуглевод\w*", r"\bпротеин\w*", r"\bпохуд\w*",
        r"\bздоров\w*", r"\bмышц\w*", r"\bмасс\w*", r"\bрацион\w*", r"\bвитамин\w*",
        r"\bминерал\w*", r"\bгидрат\w*", r"\bсон\w*", r"\bотдых\w*", r"\bвосстановлен\w*",
        r"\bвеган\w*", r"\bвегетариан\w*", r"\bкето\w*", r"\bкардио\w*", r"\bhiit\b",
        r"\bсил\w*", r"\bпресс\w*", r"\bягодиц\w*", r"\bрастяжк\w*", r"\bфункционал\w*",
        r"\bсуплемент\w*", r"\bдобавк\w*", r"\bанабол\w*", r"\bжиросжиган\w*",
        r"\bгипертроф\w*", r"\bвыносливост\w*", r"\bплиометрик\w*", r"\bдинамик\w*",
        r"\bскорост\w*", r"\bфутбол\w*", r"\bбокс\w*", r"\bбег\w*", r"\bвелосипед\w*",
        r"\bплавани\w*", r"\bлыжи\w*", r"\bсоревнован\w*", r"\batлет\w*", r"\bспортсмен\w*",
        r"\bэкипировк\w*", r"\bинвентарь\w*", r"\bсостязани\w*", r"\bтурнир\w*", r"\bфизическ\w*",

        # Питание, продукты, блюда, сладости
        r"\bеда\w*", r"\bпищ\w*", r"\bпродукт\w*", r"\bрецепт\w*", r"\bменю\w*",
        r"\bфрукт\w*", r"\bовощ\w*", r"\bшоколад\w*", r"\bконфет\w*", r"\bторт\w*",
        r"\bпирож\w*", r"\bвыпеч\w*", r"\bморож\w*", r"\bпломбир\w*", r"\bдесерт\w*",
        r"\bсладост\w*", r"\bсахар\w*", r"\bсоль\w*", r"\bспец\w*", r"\bгарнир\w*",
        r"\bкулинари\w*"
    ]
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in patterns)

# =========================================
# 5. Проверка через GPT с учётом контекста
# =========================================
async def is_topic_by_gpt(user_id: str, text: str) -> bool:
    """
    Проверяем, относится ли сообщение к фитнесу, тренировкам, здоровью или питанию,
    учитывая последние сообщения (до 10) из истории пользователя.
    """
    doc = db.collection("users").document(user_id).get()
    user_data = doc.to_dict() if doc.exists else {}
    history = user_data.get("history", [])

    # Собираем до 10 последних сообщений для контекста
    history_context = "\n".join(
        [f"{msg['role']}: {msg['text']}" for msg in history[-10:]]
    )

    system_prompt = (
        "Ты профессиональный AI-тренер, консультируешь по фитнесу, тренировкам, здоровью и питанию. "
        "Отвечай только 'да' или 'нет'. "
        "Вот последние сообщения диалога:\n"
        f"{history_context}\n\n"
        "Теперь вопрос: относится ли СЛЕДУЮЩИЙ текст к теме фитнеса, тренировок, здоровья или питания?\n"
    )

    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=0,
        max_tokens=10
    )
    answer = response.choices[0].message.content.strip().lower()
    return "да" in answer

# =========================================
# 6. Проверка ограничений по здоровью
# =========================================
def is_health_restriction_question(text: str) -> bool:
    patterns = [
        r"\bне могу\b", r"\bиз-за\b", r"\bболит\b", r"\bболь\b",
        r"\bограничен\b", r"\bнет возможности\b", r"\bпроблемы со\b", r"\bс травмой\b"
    ]
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in patterns)

# =========================================
# 7. Функции whitelist/blacklist
# =========================================
def is_in_whitelist(text: str) -> bool:
    # Whitelist содержит слова и фразы, явно указывающие на фитнес, здоровье или позитивный тон.
    whitelist = [
        # Приветствия и общие фразы
        "привет", "здравствуйте", "добрый день", "доброе утро", "хай", "приветствую",
        "как дела", "спасибо",
        # Фитнес и здоровье
        "фитнес", "тренировка", "упражнения", "бег", "кардио", "силовые", "плавание",
        "йога", "стретчинг", "физкультура", "спорт", "здоровье", "диета", "питание",
        "мотивация", "прогресс", "результат", "расписание", "план",
        # Дополнительные ключевые слова
        "физическая активность", "тренир", "кроссфит", "силовые тренировки"
    ]
    text_lower = text.lower()
    return any(word in text_lower for word in whitelist)

def is_in_blacklist(text: str) -> bool:
    # Blacklist содержит темы, явно не связанные с фитнесом и здоровым образом жизни
    blacklist = [
        # Политика и экономика
        "политика", "финансы", "экономика", "инвестиции", "бизнес", "коррупция",
        # Социальные и культурные темы
        "расизм", "религия", "война", "конфликт", "скандал", "новости", "забастовка",
        # Развлечения и медиа
        "кино", "игры", "секс", "шоу", "телевидение", "мем", "юмор",
        # Прочие нерелевантные темы
        "кредит", "банки", "инфляция", "акции", "инвестиционный", "трейдинг"
    ]
    text_lower = text.lower()
    return any(word in text_lower for word in blacklist)

# =========================================
# 8. Комбинированная функция проверки тематики
# =========================================
async def is_fitness_question_combined(user_id: str, text: str) -> bool:
    # Сначала проверяем «чёрный список»
    if is_in_blacklist(text):
        return False
    # Если есть в «белом списке», пропускаем
    if is_in_whitelist(text):
        return True
    # Если упоминается ограничение по здоровью
    if is_health_restriction_question(text):
        return True
    # Если совпало по регуляркам (слова про фитнес, тренировки, здоровье, питание и т.д.)
    if is_topic_by_regex(text):
        return True
    # Если NLP-классификатор (сентимент) считает, что текст "позитивный"
    if nlp_is_fitness_topic(text):
        return True
    # Иначе спрашиваем GPT с учётом контекста
    return await is_topic_by_gpt(user_id, text)

# =========================================
# 9. Обновление истории в Firestore
# =========================================
async def update_history(user_id: str, role: str, text: str):
    user_ref = db.collection("users").document(user_id)
    doc = user_ref.get()
    if doc.exists:
        data = doc.to_dict()
        history = data.get("history", [])
    else:
        history = []
    history.append({"role": role, "text": text})
    # Храним последние 10 сообщений
    history = history[-10:]
    user_ref.update({"history": history})

# =========================================
# 10. Формирование ответа GPT (с контекстом)
# =========================================
async def ask_gpt(user_id: str, user_message: str) -> str:
    doc = db.collection("users").document(user_id).get()
    user_data = doc.to_dict() if doc.exists else {}
    params = user_data.get("params", {})
    history = user_data.get("history", [])
    
    # Формируем контекст из параметров пользователя
    params_context = ""
    if params:
        params_context = (
            f"Параметры пользователя: Пол: {params.get('пол', 'N/A')}, "
            f"Вес: {params.get('вес', 'N/A')} кг, "
            f"Рост: {params.get('рост', 'N/A')} см, "
            f"Возраст: {params.get('возраст', 'N/A')}, "
            f"Состояние здоровья: {params.get('здоровье', 'N/A')}, "
            f"Цель: {params.get('цель', 'N/A')}."
        )
    
    # Собираем до 10 последних сообщений для контекста
    history_context = ""
    if history:
        history_context = "\n".join(
            [f"{msg['role']}: {msg['text']}" for msg in history[-10:]]
        )
    
    # Обновлённый system-промпт для дружелюбного и человечного ответа
    system_message = (
        "Ты профессиональный AI-тренер, консультируешь по фитнесу, тренировкам, здоровью и питанию. "
        f"{params_context} "
        "Если пользователь указывает, что по состоянию здоровья он не может выполнять определённые упражнения, "
        "предлагай альтернативные варианты. "
        "Отвечай дружелюбно и понятно, используя Markdown, совместимый с Telegram. "
        "Не используй заголовки вида '###'; вместо этого используй жирный текст. "
        "Если вопрос касается питания (включая любые продукты, десерты, сладости и т.д.), давай советы по калориям, балансу, "
        "умеренности и возможным альтернативам. "
        "Если вопрос не по теме, отвечай: 'Извини, я могу отвечать только на вопросы о фитнесе, тренировках и здоровом образе жизни.'"
    )
    
    # Формируем список сообщений для GPT
    messages = []
    if history_context:
        messages.append({"role": "system", "content": system_message + "\nИстория:\n" + history_context})
    else:
        messages.append({"role": "system", "content": system_message})
    
    messages.append({"role": "user", "content": user_message})
    
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.5,
        max_tokens=1000
    )
    return response.choices[0].message.content

# =========================================
# FSM для пошагового сбора параметров
# =========================================
class Onboarding(StatesGroup):
    waiting_for_gender = State()
    waiting_for_weight = State()
    waiting_for_height = State()
    waiting_for_age = State()
    waiting_for_health = State()
    waiting_for_goal = State()

# =========================================
# Хендлер для приветствий с использованием fuzzy matching
# =========================================
@dp.message(lambda msg: is_greeting_fuzzy(msg.text))
async def greet_user(message: types.Message):
    await message.answer("Привет! Чем могу помочь по фитнесу, питанию и здоровому образу жизни?")

# =========================================
# 11. Стартовая команда
# =========================================
@dp.message(CommandStart())
async def start(message: types.Message, state: FSMContext):
    user_id = str(message.from_user.id)
    user_ref = db.collection("users").document(user_id)
    doc = user_ref.get()

    if not doc.exists or not doc.to_dict().get("params"):
        # Создаём документ, если не существует
        user_ref.set({
            "name": message.from_user.full_name,
            "telegram_id": user_id,
            "subscription": "free",
            "params": {}
        }, merge=True)

        await message.answer(
            "Привет! Чтобы я мог давать персональные рекомендации, нужно задать несколько вопросов.\n"
            "Для начала, укажи свой **пол** (например: мужчина или женщина).",
            parse_mode=ParseMode.MARKDOWN
        )
        await state.set_state(Onboarding.waiting_for_gender)
    else:
        await message.answer(
            "Параметры уже заданы. Можешь задать вопрос по фитнесу и питанию!\n"
            "Если хочешь изменить цель, напиши: «поменяй мою цель на ...»",
            parse_mode=ParseMode.MARKDOWN
        )

# =========================================
# 12. Сбор параметров пошагово
# =========================================
@dp.message(Onboarding.waiting_for_gender)
async def process_gender(message: types.Message, state: FSMContext):
    gender = message.text.strip()
    await state.update_data(gender=gender)
    await message.answer("Отлично! Теперь укажи свой **вес** (кг).", parse_mode=ParseMode.MARKDOWN)
    await state.set_state(Onboarding.waiting_for_weight)

@dp.message(Onboarding.waiting_for_weight)
async def process_weight(message: types.Message, state: FSMContext):
    weight = message.text.strip()
    await state.update_data(weight=weight)
    await message.answer("Принято. Теперь укажи свой **рост** (см).", parse_mode=ParseMode.MARKDOWN)
    await state.set_state(Onboarding.waiting_for_height)

@dp.message(Onboarding.waiting_for_height)
async def process_height(message: types.Message, state: FSMContext):
    height = message.text.strip()
    await state.update_data(height=height)
    await message.answer("Понял. Теперь укажи свой **возраст** (лет).", parse_mode=ParseMode.MARKDOWN)
    await state.set_state(Onboarding.waiting_for_age)

@dp.message(Onboarding.waiting_for_age)
async def process_age(message: types.Message, state: FSMContext):
    age = message.text.strip()
    await state.update_data(age=age)
    await message.answer(
        "Есть ли у тебя какие-то **ограничения по здоровью**?\n"
        "(Например, «нет ограничений» или «болит колено»)",
        parse_mode=ParseMode.MARKDOWN
    )
    await state.set_state(Onboarding.waiting_for_health)

@dp.message(Onboarding.waiting_for_health)
async def process_health(message: types.Message, state: FSMContext):
    health = message.text.strip()
    await state.update_data(health=health)
    await message.answer("И наконец, какая у тебя **цель**? (например: похудение, набор массы и т.д.)", parse_mode=ParseMode.MARKDOWN)
    await state.set_state(Onboarding.waiting_for_goal)

@dp.message(Onboarding.waiting_for_goal)
async def process_goal(message: types.Message, state: FSMContext):
    goal = message.text.strip()
    await state.update_data(goal=goal)

    data = await state.get_data()
    gender = data["gender"]
    weight = data["weight"]
    height = data["height"]
    age = data["age"]
    health = data["health"]
    goal = data["goal"]

    user_id = str(message.from_user.id)
    params = {
        "пол": gender,
        "вес": weight,
        "рост": height,
        "возраст": age,
        "здоровье": health,
        "цель": goal
    }
    db.collection("users").document(user_id).update({"params": params})

    await message.answer(
        "Отлично! Я записал твои параметры:\n"
        f"• Пол: {gender}\n"
        f"• Вес: {weight}\n"
        f"• Рост: {height}\n"
        f"• Возраст: {age}\n"
        f"• Здоровье: {health}\n"
        f"• Цель: {goal}\n\n"
        "Теперь можешь задать вопрос по фитнесу, питанию и т.д.!",
        parse_mode=ParseMode.MARKDOWN
    )
    await state.clear()

# =========================================
# 13. Обновление цели
# =========================================
@dp.message(lambda msg: "поменяй мою цель" in msg.text.lower() or "измени мою цель" in msg.text.lower())
async def update_goal(message: types.Message):
    text_lower = message.text.lower()
    if "на" in text_lower:
        new_goal = message.text.split("на", 1)[1].strip()
        if new_goal:
            user_id = str(message.from_user.id)
            db.collection("users").document(user_id).update({"params.цель": new_goal})
            await message.answer(f"Цель обновлена на: *{new_goal}*", parse_mode=ParseMode.MARKDOWN)
            return
    await message.answer("Пожалуйста, укажи новую цель после фразы 'поменяй мою цель на'.", parse_mode=ParseMode.MARKDOWN)

# =========================================
# 14. Основной обработчик сообщений
# =========================================
@dp.message(lambda msg: not ("поменяй мою цель" in msg.text.lower() or "измени мою цель" in msg.text.lower()))
async def handle_message(message: types.Message):
    user_id = str(message.from_user.id)
    doc = db.collection("users").document(user_id).get()
    user_data = doc.to_dict() if doc.exists else {}
    params = user_data.get("params", {})

    # Если параметры не заданы, просим пройти онбординг
    if not params:
        await message.answer(
            "Чтобы я мог давать персональные рекомендации, пожалуйста, ответь на вопросы:\n"
            "пол, вес, рост, возраст, состояние здоровья, цель\n\n"
            "Или просто перезапусти /start, чтобы начать пошаговый опрос.",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    # Фильтрация вопроса
    if not await is_fitness_question_combined(user_id, message.text):
        await message.answer(
            "Прости, но я не смогу помочь с этим вопросом.\n\n"
            "Я специализируюсь на фитнесе, тренировках, питании и здоровом образе жизни.",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    await message.chat.do("typing")
    response = await ask_gpt(user_id, message.text)
    clean_response = fix_markdown_telegram(response)
    await send_split_message(message.chat.id, clean_response, parse_mode=ParseMode.MARKDOWN)

    # Обновляем историю (сохраняем вопрос и ответ)
    await update_history(user_id, "user", message.text)
    await update_history(user_id, "bot", response)

# =========================================
# 15. Запуск бота
# =========================================
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
