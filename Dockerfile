# Single image: builds the dashboard, then serves API + WebSocket + dashboard from FastAPI.
# Listens on $PORT (set by Render and similar hosts), default 8053.
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PORT=8053
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ backend/
COPY tests/ tests/
COPY --from=web /web/dist frontend/dist
RUN mkdir -p data/sample data/uploads data/benchmarks \
 && python -m backend.tools.export_sample --preset mixed_traffic --frames 10 --beams 32
EXPOSE 8053
CMD ["python", "-m", "backend", "--host", "0.0.0.0"]
