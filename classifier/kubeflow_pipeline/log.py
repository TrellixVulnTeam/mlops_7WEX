import logging
import os

import requests
from urllib.parse import parse_qs
import kfp

USERNAME = "user@example.com"
PASSWORD = "12341234"
NAMESPACE = "kubeflow-user-example-com"
HOST = 'http://210.123.42.43:8080'

session = requests.Session()
response = session.get(HOST)

headers = {
    "Content-Type": "application/x-www-form-urlencoded",
}

data = {"login": USERNAME, "password": PASSWORD}
session.post(response.url, headers=headers, data=data)
print(session.cookies.get_dict())
session_cookie = session.cookies.get_dict()["authservice_session"]

print(session_cookie)

client = kfp.Client(
    host=f"{HOST}/pipeline",
    cookies=f"authservice_session={session_cookie}",
    namespace=NAMESPACE
)

# print(client.list_pipelines())
print(client.create_experiment(name='fdsaf',namespace='kubeflow-user-example-com'))

