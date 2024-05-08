FROM python:3.10

WORKDIR /tg_bot

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . /tg_bot

CMD /start_bot.sh