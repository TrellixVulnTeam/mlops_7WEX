FROM python:3.7
COPY . /app
WORKDIR /app
RUN pip install -U numpy faiss-cpu==1.6.1 ujson
ENTRYPOINT ["python", "run.py"]
