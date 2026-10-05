# Scopewatch on Nebius. Build: docker build -t scopewatch-nebius .   (not deployed)
# Run:   docker run -p 8080:8080 -e NEBIUS_API_KEY=... scopewatch-nebius
# Without a key everything works except the live "Run it" button; cached answers still show.
FROM --platform=linux/amd64 python:3.13-slim-bookworm
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    OPENCV26_UPLOAD_DIR=/tmp/scopewatch-uploads
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/* && useradd --create-home --uid 10001 app
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt \
 && python -c "import cv2; assert cv2.__version__.startswith('5.'), cv2.__version__"
COPY backend ./backend
COPY web ./web
USER app
WORKDIR /app/backend
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD curl -fsS http://127.0.0.1:8080/healthz || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
