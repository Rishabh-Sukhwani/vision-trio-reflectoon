import base64
from openai import OpenAI

client = OpenAI()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def find_similar_face(image_path):
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

    print(response.choices[0].message.content)
