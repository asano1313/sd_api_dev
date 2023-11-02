FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

WORKDIR /var/www/app

# COPY ./pyproject.toml /var/www/app/pyproject.toml
# COPY ./poetry.lock /var/www/app/poetry.lock

RUN apt update -y
RUN apt-get update -y
RUN apt-get install libgl1-mesa-glx -y
RUN apt-get install libglib2.0-dev -y
RUN apt install -y python3.10
RUN apt install -y python3-pip
RUN pip install --upgrade pip
# RUN pip install poetry
# RUN poetry install --no-root
RUN pip install diffusers uvicorn requests fastapi invisible_watermark transformers accelerate safetensors Gunicorn

COPY ./src /var/www/app/src

CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "src.main:app", "--bind", "0.0.0.0:8080", "--timeout", "240"]