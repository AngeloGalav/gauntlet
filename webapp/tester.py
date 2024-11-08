import requests

# server url
url = 'http://localhost:5000/process-image'

image_path = 'test_image.jpg'

# Send image in a POST request
with open(image_path, 'rb') as img_file:
    response = requests.post(url, files={'image': img_file})

# Save the received plot if the request was successful
if response.status_code == 200:
    with open('result.png', 'wb') as f:
        f.write(response.content)
else:
    print("Error:", response.json())
