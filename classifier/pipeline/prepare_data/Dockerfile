FROM python:3.7

RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
COPY . /app
WORKDIR /app
RUN pip install -U numpy pandas scikit-learn pillow

ENTRYPOINT ["python" , "run.py"]

