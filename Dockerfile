FROM pytorch/pytorch:latest

WORKDIR /fedors_bot

COPY requirements.txt ./
RUN pip install --requirement requirements.txt

COPY . .

ENV BOT_KEY=6883730058:AAEkLDCTDdWWugYFG0JmBef2LsS68n19hK0

RUN python3 update_bot_models.py
RUN cd /fedors_bot/tgbot/
WORKDIR /fedors_bot/tgbot
CMD python3 bot.py
