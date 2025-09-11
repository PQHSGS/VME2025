import requests

url = "http://127.0.0.1:5000/transcribe"
data = {"data": "Hello from local machine!"}

response = requests.post(url, json=data)
print(response.json())
