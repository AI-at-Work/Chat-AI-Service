FROM python:3.9-slim

RUN apt-get update && apt-get install -y git

WORKDIR /ai-service

ADD . .

RUN pip install -r requirements.txt

CMD ["python", "main.py"]