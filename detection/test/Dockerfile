FROM pytorch/pytorch:latest

RUN pip install -U numpy albumentations tqdm mlflow boto3 timm
RUN mkdir -p /app

ARG DISABEL_CACHE
ADD . /app/

WORKDIR /app
ENTRYPOINT ["python", "test.py"]
