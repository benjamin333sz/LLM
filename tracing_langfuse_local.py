from dotenv import load_dotenv
load_dotenv()

import base64
from openai import OpenAI
from langfuse import get_client

image_path = "invoice/invoice_2.jpeg"
with open(image_path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

langfuse = get_client()

prompt = """
Extrais les informations de cette facture.
"""

schema = {
    "name": "facture",
    "schema": {
        "type": "object",
        "properties": {
            "total_ttc": {"type": ["number", "null"]},
            "total_ht": {"type": ["number", "null"]},
            "tva": {"type": ["number", "null"]},
            "devise": {
                "type": ["string", "null"],
                "description": "Code ISO 4217 sur 3 lettres"
            }
        },
        "required": ["total_ttc", "total_ht", "tva", "devise"],
        "additionalProperties": False
    }
}

with langfuse.start_as_current_observation(
    as_type="generation",
    name="ministral-facture",
    model="mistralai/ministral-3-3b",
    input=prompt
) as gen:

    response = client.chat.completions.create(
        model="mistralai/ministral-3-3b",
        temperature=0.1,
        response_format={
            "type": "json_schema",
            "json_schema": schema
        },
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ]
    )

    output = response.choices[0].message.content
    gen.update(output=output, metadata={"backend": "lm-studio-local"})

langfuse.flush()
