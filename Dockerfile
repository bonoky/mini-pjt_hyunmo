# TechScout — 클린 환경 재현용 이미지
# 빌드: docker build -t techscout .
# 실행: docker run --rm -p 8000:8000 --env-file .env techscout
#       (키는 이미지에 넣지 않고 실행 시 --env-file 로만 전달)
FROM python:3.12-slim

ENV PYTHONUTF8=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY evaluation/ evaluation/
COPY data/ data/

EXPOSE 8000
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
