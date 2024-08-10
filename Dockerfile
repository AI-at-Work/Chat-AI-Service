FROM python:3.12.3-slim

RUN apt-get update && apt-get install -y git gcc g++

WORKDIR /ai-service

ADD . .

RUN pip install -r requirements.txt

CMD ["python", "main.py"]