# Single image: builds the dashboard, then serves API + dashboard from FastAPI on :8053.
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ backend/
COPY tests/ tests/
COPY --from=web /web/dist frontend/dist
RUN mkdir -p data/sample data/uploads data/benchmarks
EXPOSE 8053
CMD ["python", "-m", "backend", "--host", "0.0.0.0", "--port", "8053"]
