import asyncio
import logging
import json
from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.enums.content_type import ContentType
from aiogram.utils.keyboard import InlineKeyboardBuilder
from collections import defaultdict
from aiogram.types import (
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)
import os
from dotenv import load_dotenv
load_dotenv()

import sys
sys.path.append("..")

from classification import Model

logging.basicConfig(level=logging.INFO)

bot = Bot(token=os.getenv('BOT_KEY'))
# storage = MemoryStorage()
dp = Dispatcher()

class Form(StatesGroup):
    model_name = State()

users_config = {}
models = {}
bot_config = {}
default_model = None

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Hello! To use this bot send an image you would like to classify. You can also choose the classifier model via /config command")

@dp.message(Command("help"))
async def cmd_start(message: types.Message):
    await message.answer("Hello! To use this bot send an image you would like to classify. You can also choose the classifier model via /config command")

@dp.message(Command("config"))
async def cmd_config(message: types.Message, state: FSMContext):
    await state.set_state(Form.model_name)
    kb = []
    for model_name in models.keys():
        kb.append([KeyboardButton(text=model_name)])

    keyboard = ReplyKeyboardMarkup(keyboard=kb, 
        resize_keyboard=True, 
        input_field_placeholder="Choose model",
        one_time_keyboard=True)
    await message.answer(
        "Type in model which you would like to use:", 
        reply_markup=keyboard
        )

@dp.message(Command("cancel"))
@dp.message(F.text.casefold() == "cancel")
async def cancel_handler(message: Message, state: FSMContext) -> None:
    """
    Allow user to cancel any action
    """
    current_state = await state.get_state()
    if current_state is None:
        return

    logging.info("Cancelling state %r", current_state)
    await state.clear()
    await message.answer(
        "Cancelled.",
        reply_markup=ReplyKeyboardRemove(),
    )

@dp.message(Form.model_name)
async def process_name(message: Message, state: FSMContext) -> None:
    await state.update_data(model_name=message.text)
    ans = None
    if message.text in models.keys():
        users_config[message.from_user.id] = message.text
        ans = message.text + ' is set as your model'
    else:
        ans = 'No model named with this name is found'

    print(users_config)
    await state.clear()
    await message.answer(ans)

@dp.message(F.photo)
async def process_photo(message: types.Message):
    model_name = users_config.get(message.from_user.id, default_model)
    print(model_name)
    await message.bot.download(file=message.photo[-1].file_id, destination='test.jpg')
    await message.reply(models[model_name].inference('test.jpg'))
    os.remove('test.jpg')

async def main():
    await dp.start_polling(bot)
    with open('bot_config.json', 'w') as f:
        json.dump(bot_config, f)

if __name__ == "__main__":

    with open('bot_config.json', 'r') as f:
        bot_config = json.load(f)

    users_config = bot_config['users_config']

    for key, value in bot_config['models_config'].items():
        default_model = key
        models[key] = Model(value['model_type'], '../' + value['state_dict'], value['inference_info'])

    asyncio.run(main())