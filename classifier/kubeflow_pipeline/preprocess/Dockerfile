FROM python:3.7

RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
COPY . /app
WORKDIR /app
RUN pip install -U numpy opencv-python==4.4.0.46

ENTRYPOINT ["python" , "run.py"]

