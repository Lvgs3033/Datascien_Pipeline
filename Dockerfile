FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir xgboost joblib

COPY app.py .
COPY templates/ templates/
COPY static/ static/

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 5000

ENV FLASK_APP=app.py

CMD ["python", "app.py"]