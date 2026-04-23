# MembraneEnv — HF Space / local Docker (stub)
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt openenv.yaml ./
COPY models.py client.py ./
COPY server ./server
COPY tasks ./tasks

RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 7860
ENV PYTHONPATH=/app
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
