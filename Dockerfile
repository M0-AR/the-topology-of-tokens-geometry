FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.13.0

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TOPOBPE_ROOT=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md requirements.lock ./
COPY src ./src
RUN pip install --upgrade pip \
    && pip install --index-url "${TORCH_INDEX}" "torch==${TORCH_VERSION}" \
    && pip install -r requirements.lock \
    && pip install --no-deps -e .

COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY docs ./docs

ENTRYPOINT ["python", "-m", "topobpe"]
CMD ["--help"]
