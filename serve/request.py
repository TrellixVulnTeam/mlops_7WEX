import requests
url="http://{your_ip}:{your_port}/predict"
test_files = {
    "test_file_1": open("{image_path}", "rb")
}
response = requests.post(url, files=test_files)
print(response.json)

