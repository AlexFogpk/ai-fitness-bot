import logging
import asyncio
import os
import re
import difflib  # для fuzzy matching

from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart, Command
from aiogram.enums import ParseMode
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton  # импорт для работы с клавиатурой
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
# 1. Константы окружения
# =========================================
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =========================================
# 2. Firebase инициализация
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
# 3. Инициализация OpenAI (GPT-4o-mini)
# =========================================
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# =========================================
# 4. Инициализация NLP-модели (русская модель для сентимент-анализа)
# =========================================
model_name = "cointegrated/rubert-tiny2"
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=huggingface_token)
model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=huggingface_token)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

def nlp_is_fitness_topic(text: str) -> bool:
    result = classifier(text)
    label = result[0]["label"].lower()
    return label == "positive"

# =========================================
# 5. Инициализация бота и Dispatcher
# =========================================
logging.basicConfig(level=logging.INFO)
bot = Bot(token=BOT_TOKEN)
storage = MemoryStorage()  # Храним состояние в памяти
dp = Dispatcher(storage=storage)

# =========================================
# 6. Определение Reply Keyboard для главного меню
# =========================================
btn_change_data = KeyboardButton(text="Изменить данные")
btn_change_goal = KeyboardButton(text="Изменить цель")
btn_calculate_kbju = KeyboardButton(text="Посчитать КБЖУ")
btn_plans = KeyboardButton(text="Планы тренировок")
btn_support = KeyboardButton(text="Тех. Поддержка")
btn_subscription = KeyboardButton(text="Подписка")

main_menu_kb = ReplyKeyboardMarkup(
    keyboard=[
        [btn_change_data, btn_change_goal],
        [btn_calculate_kbju, btn_plans],
        [btn_support, btn_subscription]
    ],
    resize_keyboard=True  # клавиатура будет компактной
)

# Дополнительная клавиатура для выбора уровня активности
activity_kb = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Сидячий (1.2)")],
        [KeyboardButton(text="Лёгкая активность (1.375)")],
        [KeyboardButton(text="Средняя активность (1.55)")],
        [KeyboardButton(text="Высокая активность (1.7)")],
        [KeyboardButton(text="Очень высокая (1.9)")]
    ],
    resize_keyboard=True
)

# =========================================
# 7. Функция для обработки Markdown
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
# 8. Функция разбивки длинного текста
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
# 9. Отправка сообщений по частям
# =========================================
async def send_split_message(chat_id, text, parse_mode=None):
    parts = split_message(text)
    for part in parts:
        await bot.send_message(chat_id, part, parse_mode=parse_mode)

# =========================================
# 10. Fuzzy matching для приветствий
# =========================================
GREETINGS = [
    "привет", "здравствуйте", "добрый день", "доброе утро", "хай", "приветствую",
    "здарова", "салют", "хелло", "хелоу", "хей", "хэй", "йоу",
    "hello", "hi", "hey", "good morning", "good day"
]

def is_greeting_fuzzy(text: str) -> bool:
    text_lower = text.lower().strip()
    matches = difflib.get_close_matches(text_lower, GREETINGS, n=1, cutoff=0.8)
    return len(matches) > 0

# =========================================
# 11. Фильтрация тематики (регулярки и дополнительные проверки)
# =========================================
def is_topic_by_regex(text: str) -> bool:
    patterns_fitness = [
        r"\bфитнес\w*", r"\bтрениров\w*", r"\bтренир\w*", r"\bупражн\w*",
        r"\bфизкульт\w*", r"\bспорт\w*", r"\bсил\w*", r"\bпресс\w*",
        r"\bягодиц\w*", r"\bрастяжк\w*", r"\bвыносливост\w*"
    ]
    patterns_healthy_food = [
        r"\bдиет\w*", r"\bпитан\w*", r"\bкалор\w*", r"\bбелк\w*",
        r"\bовощ\w*", r"\bфрукт\w*", r"\bменю\w*", r"\bрецепт\w*"
    ]
    patterns_unhealthy_food = [
        r"\bчипс\w*", r"\bснэк\w*", r"\bфастфуд\w*", r"\bбургер\w*", r"\bгамбургер\w*",
        r"\bшаурм\w*", r"\bдонер\w*", r"\bкартофел[ьья]\s?фри", r"\bфри\b",
        r"\bмайонез\w*", r"\bкетчуп\w*", r"\bсоус\w*", r"\bнаггетс\w*",
        r"\bпицц\w*", r"\bролл\w*", r"\bсуши\w*", r"\bхотдог\w*",
        r"\bсэндвич\w*", r"\bбутерброд\w*", r"\bджанкфуд\w*", r"\bjunk food\b",
        r"\bгазиров\w*", r"\bкол\w*", r"\bпепси\w*", r"\bспрайт\w*",
        r"\bэнергетик\w*", r"\bалкогол\w*", r"\bпиво\w*", r"\bвино\w*", r"\bспиртн\w*",
        r"\bсухар\w*",
        r"\bшоколад\w*", r"\bконфет\w*", r"\bторт\w*", r"\bпирож\w*", r"\bвыпеч\w*",
        r"\bмакарон\w*", r"\bпаста\w*", r"\bбулк\w*", r"\bхлеб\w*", r"\bбатон\w*"
    ]
    all_patterns = patterns_fitness + patterns_healthy_food + patterns_unhealthy_food
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in all_patterns)

def is_health_restriction_question(text: str) -> bool:
    patterns = [
        r"\bне могу\b", r"\bиз-за\b", r"\bболит\b", r"\bболь\b",
        r"\bограничен\b", r"\bнет возможности\b", r"\bпроблемы со\b", r"\bс травмой\b"
    ]
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in patterns)

def is_in_whitelist(text: str) -> bool:
    whitelist = [
        "привет", "здравствуйте", "добрый день", "доброе утро", "хай", "приветствую",
        "как дела", "спасибо",
        "фитнес", "тренировка", "упражнения", "бег", "кардио", "силовые", "плавание",
        "йога", "стретчинг", "физкультура", "спорт", "здоровье", "диета", "питание",
        "мотивация", "прогресс", "результат", "расписание", "план",
        "физическая активность", "тренир", "кроссфит", "силовые тренировки"
    ]
    text_lower = text.lower()
    return any(word in text_lower for word in whitelist)

def is_in_blacklist(text: str) -> bool:
    blacklist = [
        "политика", "финансы", "экономика", "инвестиции", "бизнес", "коррупция",
        "расизм", "религия", "война", "конфликт", "скандал", "новости", "забастовка",
        "кино", "игры", "секс", "шоу", "телевидение", "мем", "юмор",
        "кредит", "банки", "инфляция", "акции", "инвестиционный", "трейдинг"
    ]
    text_lower = text.lower()
    return any(word in text_lower for word in blacklist)

# =========================================
# 12. Комбинированная проверка тематики
# =========================================
async def is_fitness_question_combined(user_id: str, text: str) -> bool:
    if is_in_blacklist(text):
        return False
    if is_in_whitelist(text):
        return True
    if is_health_restriction_question(text):
        return True
    if is_topic_by_regex(text):
        return True
    if nlp_is_fitness_topic(text):
        return True
    return await is_topic_by_gpt(user_id, text)

# =========================================
# 13. GPT fallback для проверки тематики
# =========================================
async def is_topic_by_gpt(user_id: str, text: str) -> bool:
    doc = db.collection("users").document(user_id).get()
    user_data = doc.to_dict() if doc.exists else {}
    history = user_data.get("history", [])
    history_context = "\n".join([f"{msg['role']}: {msg['text']}" for msg in history[-10:]])
    system_prompt = (
        "Ты эксперт по фитнесу, тренировкам, здоровью и питанию. Отвечай только 'да' или 'нет'.\n"
        f"История диалога:\n{history_context}\n\n"
        "Относится ли следующий текст к теме фитнеса, тренировок, здоровью или питанию?\n"
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
# 14. Обновление истории в Firestore
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
    history = history[-5:]
    user_ref.update({"history": history})

# =========================================
# 15. Формирование ответа GPT (с контекстом)
# =========================================
async def ask_gpt(user_id: str, user_message: str) -> str:
    doc = db.collection("users").document(user_id).get()
    user_data = doc.to_dict() if doc.exists else {}
    params = user_data.get("params", {})
    history = user_data.get("history", [])
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
    history_context = ""
    if history:
        history_context = "\n".join([f"{msg['role']}: {msg['text']}" for msg in history[-10:]])
    system_message = (
        "Ты профессиональный AI-тренер, консультируешь по фитнесу, здоровью и питанию. "
        f"{params_context} "
        "Если пользователь спрашивает про вредные продукты (чипсы, фастфуд, алкоголь и т.д.), "
        "объясняй возможный вред, указывай калорийность, давай советы по умеренности и предлагай более здоровые альтернативы. "
        "Если пользователь спрашивает про здоровое питание, тренировки, баланс — помогай. "
        "Если есть ограничения по здоровью, предлагай альтернативы. "
        "Отвечай дружелюбно и понятно, используя Markdown, совместимый с Telegram. "
        "Не используй заголовки вида '###'; вместо этого используй жирный текст. "
        "Если вопрос не по теме, отвечай: 'Извини, я могу отвечать только на вопросы о фитнесе, тренировках и здоровом образе жизни.'"
    )
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
# 16. FSM для сбора параметров (онбординг)
# =========================================
class Onboarding(StatesGroup):
    waiting_for_gender = State()
    waiting_for_weight = State()
    waiting_for_height = State()
    waiting_for_age = State()
    waiting_for_health = State()
    waiting_for_goal = State()
    waiting_for_activity = State()  # новый шаг для выбора активности

# =========================================
# 17. FSM для изменения цели (короткий опрос)
# =========================================
class ChangeGoal(StatesGroup):
    waiting_for_new_goal = State()

# =========================================
# 18. Хендлер для приветствий (fuzzy matching)
# =========================================
@dp.message(lambda msg: is_greeting_fuzzy(msg.text))
async def greet_user(message: types.Message):
    await message.answer("Привет! Чем могу помочь по фитнесу, питанию и здоровому образу жизни?")

# =========================================
# 19. Стартовая команда с показом клавиатуры
# =========================================
@dp.message(CommandStart())
async def start(message: types.Message, state: FSMContext):
    user_id = str(message.from_user.id)
    user_ref = db.collection("users").document(user_id)
    doc = user_ref.get()
    if not doc.exists or not doc.to_dict().get("params"):
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
        # Если данные уже есть, отправляем клавиатуру с главным меню
        await message.answer(
            "Параметры уже заданы. Можешь выбрать действие в меню:",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=main_menu_kb
        )

#########################
# 20. Логика для кнопок меню
#########################

# 20.1. Изменить данные – повторный опрос
@dp.message(lambda msg: msg.text == "Изменить данные")
async def handle_change_data(message: types.Message, state: FSMContext):
    await message.answer(
        "Хорошо! Давай заново укажем параметры.\n\n"
        "Для начала, укажи свой **пол** (например: мужчина или женщина).",
        parse_mode=ParseMode.MARKDOWN
    )
    await state.set_state(Onboarding.waiting_for_gender)

# 20.2. Изменить цель – запуск короткого FSM для ввода новой цели
@dp.message(lambda msg: msg.text == "Изменить цель")
async def handle_change_goal_button(message: types.Message, state: FSMContext):
    await message.answer("Окей! Введи, пожалуйста, новую цель (например: похудение, набор массы и т.д.)")
    await state.set_state(ChangeGoal.waiting_for_new_goal)

@dp.message(ChangeGoal.waiting_for_new_goal)
async def process_new_goal(message: types.Message, state: FSMContext):
    new_goal = message.text.strip()
    user_id = str(message.from_user.id)
    db.collection("users").document(user_id).update({"params.цель": new_goal})
    await message.answer(f"Цель обновлена на: *{new_goal}*", parse_mode=ParseMode.MARKDOWN)
    await state.clear()

# 20.3. Посчитать КБЖУ – расчет калорий и макронутриентов с учётом активности и цели
@dp.message(lambda msg: msg.text == "Посчитать КБЖУ")
async def handle_calculate_kbju(message: types.Message):
    user_id = str(message.from_user.id)
    doc = db.collection("users").document(user_id).get()
    user_data = doc.to_dict() if doc.exists else {}

    if not user_data or "params" not in user_data:
        await message.answer(
            "Чтобы рассчитать КБЖУ, мне нужны твои параметры. Пожалуйста, сначала задай их с помощью /start или 'Изменить данные'."
        )
        return

    params = user_data["params"]
    gender = params.get("пол", "").lower()
    try:
        weight = float(params.get("вес", 0))
        height = float(params.get("рост", 0))
        age = float(params.get("возраст", 0))
    except ValueError:
        await message.answer("Некорректные данные. Пожалуйста, обнови свои параметры через 'Изменить данные'.")
        return

    if not (weight > 0 and height > 0 and age > 0 and (gender in ["мужчина", "женщина"])):
        await message.answer("Похоже, твои параметры неполные или некорректные. Попробуй 'Изменить данные'.")
        return

    activity_factor = float(params.get("активность", 1.375))
    # Расчет BMR по формуле Миффлина – Сан Жеора
    if gender == "мужчина":
        bmr = 9.99 * weight + 6.25 * height - 4.92 * age + 5
    else:
        bmr = 9.99 * weight + 6.25 * height - 4.92 * age - 161

    tdee = bmr * activity_factor

    # Корректировка TDEE в зависимости от цели
    goal_lower = params.get("цель", "").lower()
    if "похуд" in goal_lower:
        factor_goal = 0.85  # вычесть примерно 15%
        protein_factor = 1.8
    elif "набор" in goal_lower:
        factor_goal = 1.15  # прибавить примерно 15%
        protein_factor = 1.5
    else:
        factor_goal = 1.0
        protein_factor = 1.5

    tdee_adjusted = tdee * factor_goal

    # Расчет макронутриентов:
    protein_g = protein_factor * weight
    fat_g = 1.0 * weight
    cals_from_protein = protein_g * 4
    cals_from_fat = fat_g * 9
    carbs_cals = tdee_adjusted - (cals_from_protein + cals_from_fat)
    carbs_g = carbs_cals / 4 if carbs_cals > 0 else 0

    response_text = (
        f"Твои расчётные показатели (приблизительно):\n\n"
        f"Суточная потребность в калориях: ~{int(tdee_adjusted)} ккал\n\n"
        f"Белки: {int(protein_g)} г/день\n"
        f"Жиры: {int(fat_g)} г/день\n"
        f"Углеводы: {int(carbs_g)} г/день\n\n"
        f"Учти, что это приблизительный расчёт, скорректированный с учётом твоей активности и цели ({params.get('цель', 'N/A')})."
    )
    await message.answer(response_text)

#########################
# 21. Сбор параметров пошагово (онбординг)
#########################
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

# После ввода цели просим выбрать уровень активности
@dp.message(Onboarding.waiting_for_goal)
async def process_goal(message: types.Message, state: FSMContext):
    goal = message.text.strip()
    await state.update_data(goal=goal)
    await message.answer(
        "Выбери уровень физической активности:",
        reply_markup=activity_kb,
        parse_mode=ParseMode.MARKDOWN
    )
    await state.set_state(Onboarding.waiting_for_activity)

# Обработка выбора активности
@dp.message(Onboarding.waiting_for_activity)
async def process_activity(message: types.Message, state: FSMContext):
    activity_text = message.text.strip()
    match = re.search(r"\(([\d\.]+)\)", activity_text)
    if match:
        activity_factor = float(match.group(1))
    else:
        activity_factor = 1.2
    await state.update_data(activity=activity_factor)
    data = await state.get_data()
    user_id = str(message.from_user.id)
    params = {
        "пол": data.get("gender"),
        "вес": data.get("weight"),
        "рост": data.get("height"),
        "возраст": data.get("age"),
        "здоровье": data.get("health"),
        "цель": data.get("goal"),
        "активность": activity_factor
    }
    db.collection("users").document(user_id).update({"params": params})
    await message.answer(
        "Отлично! Я записал твои параметры:\n"
        f"• Пол: {data.get('gender')}\n"
        f"• Вес: {data.get('weight')}\n"
        f"• Рост: {data.get('height')}\n"
        f"• Возраст: {data.get('age')}\n"
        f"• Здоровье: {data.get('health')}\n"
        f"• Цель: {data.get('goal')}\n"
        f"• Активность: {activity_factor}\n\n"
        "Теперь можешь задать вопрос по фитнесу, питанию и т.д.!",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_menu_kb
    )
    await state.clear()

# =========================================
# 22. Обновление цели через текстовую фразу (альтернативный вариант)
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
# 23. Основной обработчик сообщений (общий fallback)
# =========================================
@dp.message(lambda msg: not ("поменяй мою цель" in msg.text.lower() or "измени мою цель" in msg.text.lower()))
async def handle_message(message: types.Message):
    user_id = str(message.from_user.id)
    doc = db.collection("users").document(user_id).get()
    user_data = doc.to_dict() if doc.exists else {}
    params = user_data.get("params", {})
    if not params:
        await message.answer(
            "Чтобы я мог давать персональные рекомендации, пожалуйста, ответь на вопросы:\n"
            "пол, вес, рост, возраст, состояние здоровья, цель\n\n"
            "Или просто перезапусти /start, чтобы начать пошаговый опрос.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
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
    await update_history(user_id, "user", message.text)
    await update_history(user_id, "bot", response)

# =========================================
# 24. Точка входа
# =========================================
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
