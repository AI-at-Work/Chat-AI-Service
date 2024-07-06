FROM python:3.9-slim

WORKDIR /ai-service

ADD . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]