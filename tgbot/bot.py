import asyncio
import logging
import json
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters.command import Command
from aiogram.enums.content_type import ContentType
from collections import defaultdict
import os
import sys
sys.path.append("..")

from classification import Model

logging.basicConfig(level=logging.INFO)
bot = Bot(token="6883730058:AAEkLDCTDdWWugYFG0JmBef2LsS68n19hK0")
dp = Dispatcher()

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Hello!")

model = None

@dp.message(F.photo)
async def process_photo(message: types.Message):
    await message.bot.download(file=message.photo[-1].file_id, destination='test.jpg')
    await message.answer(model.inference('test.jpg'))
    os.remove('test.jpg')

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":

    bot_config = None
    with open('bot_config.json', 'r') as f:
        bot_config = json.load(f)

    # info = {"classes": ["Ill_cucumber", "good_Cucumber"], "valid_transforms": {}}
    # model = Model('QResNet18', '/home/fedor/Coursework2024/models/ResNet18_quantized_model', info)
    print(bot_config)
    asyncio.run(main())