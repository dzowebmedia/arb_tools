FROM python:3.12-bookworm

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python -m playwright install --with-deps chromium

COPY . .

ENV PORT=8080
CMD ["sh","-c","uvicorn service:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers --forwarded-allow-ips='*'"]
