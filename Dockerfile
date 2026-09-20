FROM python:3.11-slim

WORKDIR /app

# Dependencies first so code edits do not bust the layer cache.
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

COPY src/ ./src/
COPY data/documents/ ./data/documents/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
