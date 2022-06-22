import requests

with open("/home/jongjin/st/dataset/animals_resize/cats_00001.jpg", "rb") as f:
    image_bytes = f.read()  # from file path

files = {
    "image": ("test.jpg", image_bytes),
}
response = requests.post('http://210.123.42.41:5555', files=files)
print(response)
