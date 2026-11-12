# AutoPilot-ML: the dashboard (port 8501) and the local HTTP API (port 8000) in one image (optional task 35).
#
#   docker build -t autopilot-ml:0.1.0 .
#   docker run --rm -p 127.0.0.1:8501:8501 -p 127.0.0.1:8000:8000 \
#       -v autopilot-experiments:/app/experiments -v autopilot-models:/app/models -v autopilot-mlruns:/app/mlruns \
#       autopilot-ml:0.1.0
#
# Publish the ports on 127.0.0.1 only: the API has no login. Inside the container the servers listen on
# 0.0.0.0, because that is the only way to reach them through the published ports.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# LightGBM and XGBoost need the OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY scripts/ scripts/
COPY research/ research/
COPY data/sample/ data/sample/

RUN useradd --create-home --uid 1000 autopilot \
    && mkdir -p experiments models mlruns data/raw data/processed \
    && chown -R autopilot:autopilot /app
USER autopilot

EXPOSE 8501 8000
VOLUME ["/app/experiments", "/app/models", "/app/mlruns"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["sh", "scripts/docker_entry.sh"]
