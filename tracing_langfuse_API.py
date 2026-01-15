from dotenv import load_dotenv
load_dotenv()

import json
from google import genai
from google.genai import types
from langfuse import get_client

# Init clients
langfuse = get_client()
genai_client = genai.Client()

def analyse_facture(image_path: str):
    model = "gemini-2.5-flash"

    # Charger l'image
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    prompt = ("""
    Tu es une API.
    Tu DOIS répondre UNIQUEMENT avec un JSON valide.
    Aucun texte, aucun commentaire, aucun markdown.

    Format EXACT :
    {
    "total_ttc": number,
    "total_ht": number,
    "tva": number,
    "devise": string
    }
    
    Règles importantes :
    - La devise DOIT être au format ISO 4217 sur 3 lettres (ex: EUR, USD, GBP).
    - Si une valeur est inconnue, mets null.
    """
)

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg"
                ),
                types.Part.from_text(text=prompt),
            ],
        )
    ]

    # 🔍 TRACE PRINCIPALE
    with langfuse.start_as_current_observation(
        as_type="span",
        name="analyse-facture2",
        metadata={"image": image_path}
    ) as span:

        # 🤖 GENERATION LLM
        with langfuse.start_as_current_observation(
            as_type="generation",
            name="gemini-ocr-facture",
            model=model,
            input=prompt
        ) as generation:

            response = genai_client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=400,
                    response_mime_type="application/json",
                )
            )

            raw_output = response.text.strip()
            generation.update(output=raw_output)

        # Tentative de parsing JSON
        try:
            data = json.loads(raw_output)
            span.update(
                output=data,
                metadata={"status": "success"}
            )
            return data

        except json.JSONDecodeError:
            span.update(
                output=raw_output,
                metadata={"status": "json_error"}
            )
            raise ValueError("Réponse non JSON")

if __name__ == "__main__":
    result = analyse_facture("invoice/invoice_2.jpeg")
    print(result)
    langfuse.flush()
