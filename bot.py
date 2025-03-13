from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart, Command
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
import logging
import asyncio
import os
import re
import firebase_admin
from firebase_admin import credentials, firestore
from openai import AsyncOpenAI

# =========================================
# 1. Инициализация и конфигурация
# =========================================
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

firebase_credentials = os.getenv("FIREBASE_CREDENTIALS")
if firebase_credentials:
    with open("firebase.json", "w") as f:
        f.write(firebase_credentials)
else:
    print("Внимание: переменная FIREBASE_CREDENTIALS не установлена!")

cred = credentials.Certificate("firebase.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

bot = Bot(token=BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)
logging.basicConfig(level=logging.INFO)

# =========================================
# 2. Вспомогательные функции для отправки и форматирования сообщений
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

async def send_split_message(chat_id, text, parse_mode=None):
    parts = split_message(text)
    for part in parts:
        await bot.send_message(chat_id, part, parse_mode=parse_mode)

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
# 3. Фильтрация запросов: регулярки, whitelist, blacklist, проверка ограничений по здоровью
# =========================================
def is_topic_by_regex(text: str) -> bool:
    patterns = [
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
        r"\bэкипировк\w*", r"\bинвентарь\w*", r"\bсостязани\w*", r"\bтурнир\w*", r"\bфизическ\w*"
    ]
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in patterns)

async def is_topic_by_gpt(text: str) -> bool:
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ты эксперт по фитнесу, тренировкам, здоровью и питанию. Отвечай только 'да' или 'нет'."},
            {"role": "user", "content": f"Относится ли следующий текст к теме фитнеса, тренировкам, здоровью или питанию? Текст: {text}"}
        ],
        temperature=0,
        max_tokens=10
    )
    answer = response.choices[0].message.content.strip().lower()
    return "да" in answer

def is_health_restriction_question(text: str) -> bool:
    patterns = [
        r"\bне могу\b", r"\bиз-за\b", r"\bболит\b", r"\bболь\b",
        r"\bограничен\b", r"\bнет возможности\b", r"\bпроблемы со\b", r"\bс травмой\b"
    ]
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in patterns)

def is_in_whitelist(text: str) -> bool:
    whitelist = [
        "привет", "здравствуйте", "как дела", "спасибо",
        "погода", "прогноз", "температура", "маршрут", "пробежк"
    ]
    text_lower = text.lower()
    return any(w in text_lower for w in whitelist)

def is_in_blacklist(text: str) -> bool:
    blacklist = [
        "политика", "финансы", "расизм", "кредит"
    ]
    text_lower = text.lower()
    return any(b in text_lower for b in blacklist)

async def is_fitness_question_combined(text: str) -> bool:
    if is_in_blacklist(text):
        return False
    if is_in_whitelist(text):
        return True
    if is_health_restriction_question(text):
        return True
    if is_topic_by_regex(text):
        return True
    return await is_topic_by_gpt(text)

# =========================================
# 4. Работа с историей переписки (контекст для GPT)
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
    # Сохраним последние 10 сообщений
    history = history[-10:]
    user_ref.update({"history": history})

# =========================================
# 5. Формирование ответа от GPT с расширенным контекстом
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
    
    # Используем последние 10 сообщений для расширенного контекста
    history_context = ""
    if history:
        history_context = "\n".join([f"{msg['role']}: {msg['text']}" for msg in history[-10:]])
    
    system_message = (
        "Ты профессиональный AI-тренер, консультируешь по фитнесу, тренировкам, здоровью и питанию. "
        f"{params_context} "
        "Если пользователь указывает, что по состоянию здоровья он не может выполнять определённые упражнения, "
        "предлагай альтернативные варианты. "
        "Отвечай кратко и структурированно, используя Markdown, совместимый с Telegram. "
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
# 6. FSM для пошагового сбора параметров (онбординг)
# =========================================
class Onboarding(StatesGroup):
    waiting_for_gender = State()
    waiting_for_weight = State()
    waiting_for_height = State()
    waiting_for_age = State()
    waiting_for_health = State()
    waiting_for_goal = State()

# =========================================
# 7. Онбординг: сбор параметров через FSM
# =========================================
@dp.message_handler(CommandStart())
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
        await message.answer(
            "Параметры уже заданы. Можешь задать вопрос по фитнесу и питанию!\n"
            "Если хочешь изменить цель, напиши: «поменяй мою цель на ...»",
            parse_mode=ParseMode.MARKDOWN
        )

@dp.message_handler(state=Onboarding.waiting_for_gender)
async def process_gender(message: types.Message, state: FSMContext):
    gender = message.text.strip()
    await state.update_data(gender=gender)
    await message.answer("Отлично! Теперь укажи свой **вес** (кг).", parse_mode=ParseMode.MARKDOWN)
    await state.set_state(Onboarding.waiting_for_weight)

@dp.message_handler(state=Onboarding.waiting_for_weight)
async def process_weight(message: types.Message, state: FSMContext):
    weight = message.text.strip()
    await state.update_data(weight=weight)
    await message.answer("Принято. Теперь укажи свой **рост** (см).", parse_mode=ParseMode.MARKDOWN)
    await state.set_state(Onboarding.waiting_for_height)

@dp.message_handler(state=Onboarding.waiting_for_height)
async def process_height(message: types.Message, state: FSMContext):
    height = message.text.strip()
    await state.update_data(height=height)
    await message.answer("Понял. Теперь укажи свой **возраст** (лет).", parse_mode=ParseMode.MARKDOWN)
    await state.set_state(Onboarding.waiting_for_age)

@dp.message_handler(state=Onboarding.waiting_for_age)
async def process_age(message: types.Message, state: FSMContext):
    age = message.text.strip()
    await state.update_data(age=age)
    await message.answer(
        "Есть ли у тебя какие-то **ограничения по здоровью**?\n"
        "(Например, «нет ограничений» или «болит колено»)",
        parse_mode=ParseMode.MARKDOWN
    )
    await state.set_state(Onboarding.waiting_for_health)

@dp.message_handler(state=Onboarding.waiting_for_health)
async def process_health(message: types.Message, state: FSMContext):
    health = message.text.strip()
    await state.update_data(health=health)
    await message.answer("И наконец, какая у тебя **цель**? (например: похудение, набор массы и т.д.)", parse_mode=ParseMode.MARKDOWN)
    await state.set_state(Onboarding.waiting_for_goal)

@dp.message_handler(state=Onboarding.waiting_for_goal)
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
# 8. Обновление цели по запросу
# =========================================
@dp.message_handler(lambda msg: "поменяй мою цель" in msg.text.lower() or "измени мою цель" in msg.text.lower())
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
# 9. Основной обработчик сообщений (исключая обновление цели)
# =========================================
@dp.message_handler(lambda msg: not ("поменяй мою цель" in msg.text.lower() or "измени мою цель" in msg.text.lower()))
async def handle_message(message: types.Message):
    user_id = str(message.from_user.id)
    doc = db.collection("users").document(user_id).get()
    user_data = doc.to_dict() if doc.exists else {}
    params = user_data.get("params", {})

    if not params:
        await message.answer(
            "Чтобы я мог давать персональные рекомендации, пожалуйста, ответь на вопросы:\n"
            "пол, вес, рост, возраст, состояние здоровья, цель\n\n"
            "Или перезапусти /start, чтобы пройти опрос.",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    if not await is_fitness_question_combined(message.text):
        await message.answer(
            "Прости, но я не смогу помочь с этим вопросом.\n\n"
            "Я специализируюсь на фитнесе, тренировках, питании, здоровом образе жизни.",
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
# 10. Запуск бота
# =========================================
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
