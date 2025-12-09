FROM python:3.11-slim as builder

ENV PYTHONUNBUFFERED 1
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libblas-dev \
    liblapack-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /WIKED

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

FROM python:3.11-slim as final

ENV PYTHONUNBUFFERED 1
ENV PORT 8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]