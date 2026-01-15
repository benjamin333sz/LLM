# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
from google import genai
from google.genai import types


def generate(image):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-flash"
    with open(image, "rb") as f:
            image_bytes = f.read()

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg"  # ou image/png
                ),
                types.Part.from_text(
                text=(
                    "Analyse cette facture et renvoie obligatoirement un JSON avec le format suivant: "
                    "{"
                    "'total_ttc': X, "
                    "'total_ht': X, "
                    "'tva': X ,"
                    "'devise': XXX,"
                    "}"
                )
            )
            ],
        ),
    ]
    tools = [
        types.Tool(googleSearch=types.GoogleSearch(
        )),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=-1,
        ),
        tools=tools,
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")

if __name__ == "__main__":
    for i in range(1,12):
        generate(f"invoice_{i}.jpeg")
