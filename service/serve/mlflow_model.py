import torch
import os
import mlflow
from mlflow.tracking import MlflowClient
def load_model(model_name, version):
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://210.123.42.41:9090"
    client = MlflowClient("http://210.123.42.41:5000")

    filter_string = "name='{}'".format(model_name)
    results = client.search_model_versions(filter_string)  # 버전별로 묶여나옴
    for res in results:
        if res.version == str(version):
            model_uri = res.source
            break
    reconstructed_model = mlflow.pytorch.load_model(model_uri)
    return reconstructed_model

