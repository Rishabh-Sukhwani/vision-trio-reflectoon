import base64
from openai import OpenAI

client = OpenAI()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Path to your image
#image_path = r"C:\Users\risha\OneDrive\Desktop\Rishabh\projects\hack\input_image_5.jpg"
#image_path = r"C:\Users\risha\OneDrive\Pictures\Camera Roll\WIN_20250203_11_11_51_Pro.jpg"
image_path = r"C:\Users\risha\OneDrive\Pictures\Camera Roll\WIN_20250203_11_15_24_Pro.jpg"

# Getting the Base64 string
base64_image = encode_image(image_path)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Your job is to view the input image and choose one out of the listed cartoon characters which the person in the image resembles. You MUST choose one out of the options provided.Even if the person does not look like a character out of the given options, you MUST choose the best one that the person resembles. Answer in one word from the choices provided."},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Which cartoon character (out of woody, buzz lightyear, wreck-it-ralph) does the person in the image most look like?",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ],
)

print(response.choices[0])