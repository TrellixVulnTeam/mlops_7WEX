FROM pytorch/pytorch:latest
RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libgtk2.0-dev
RUN pip install -U pandas scikit-learn numpy ujson opencv-python==4.4.0.46 faiss-cpu==1.6.1
COPY . /app
WORKDIR /app
ENTRYPOINT ["python", "run.py"]
