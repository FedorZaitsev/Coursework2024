import asyncio
import logging
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters.command import Command
from aiogram.enums.content_type import ContentType
import sys
sys.path.append("..")

from classification import Model

# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)
# Объект бота
bot = Bot(token="6883730058:AAEkLDCTDdWWugYFG0JmBef2LsS68n19hK0")
# Диспетчер
dp = Dispatcher()

# Хэндлер на команду /start
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Hello!")

model = None

@dp.message(F.photo)
async def process_photo(message: types.Message):
    await message.bot.download(file=message.photo[-1].file_id, destination = 'test.jpg')
    await message.answer(model.inference('test.jpg'))

# Запуск процесса поллинга новых апдейтов
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    info = {"classes": ["Ill_cucumber", "good_Cucumber"], "valid_transforms": {}}
    model = Model('QResNet18', '/home/fedor/Coursework2024/models/ResNet18_quantized_model', info)
    asyncio.run(main())