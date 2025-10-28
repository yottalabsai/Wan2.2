import requests

url_img = "https://fi-keyframes-test.s3-accelerate.amazonaws.com/generated/image_7ce36f39-15c4-4092-bbc3-348d5cc89971.jpg"
response_img = requests.get(url_img)
with open("./ref.jpg", "wb") as f:
    f.write(response_img.content)

# url_mp4 = ""
# response_mp4 = requests.get(url)
# with open("./ref.mp4", "wb") as f:
#     f.write(response_img.content)
