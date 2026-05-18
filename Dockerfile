ARG PYTHON_IMAGE=docker.m.daocloud.io/library/python:3.12-slim
FROM ${PYTHON_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY agentkits ./agentkits
COPY examples ./examples

RUN pip install --no-cache-dir .

CMD ["python", "examples/08_reproducible_demo.py"]
