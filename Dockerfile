FROM python:3.10

WORKDIR /tg_bot

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . /tg_bot
ENV BOT_KEY=

CMD /start_bot.sh