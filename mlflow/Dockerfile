FROM continuumio/miniconda3
RUN apt-get update && apt-get install -y git

COPY requirements.txt .
RUN pip install -r requirements.txt
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

RUN pip list
# RUN cd /home && git clone https://github.com/mlflow/mlflow-example.git
