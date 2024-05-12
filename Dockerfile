FROM pytorch/pytorch:latest

WORKDIR /fedors_bot

COPY requirements.txt ./
RUN pip install --requirement requirements.txt

COPY . .

ENV BOT_KEY=

RUN python3 update_bot_models.py
RUN cd /fedors_bot/tgbot/
WORKDIR /fedors_bot/tgbot
CMD python3 bot.py
