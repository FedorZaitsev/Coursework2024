FROM python:3.10

WORKDIR /tg_bot

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . /tg_bot
ENV BOT_KEY=6883730058:AAEkLDCTDdWWugYFG0JmBef2LsS68n19hK0

CMD /start_bot.sh