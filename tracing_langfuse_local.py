from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from langfuse import get_client

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"  # valeur dummy
)

langfuse = get_client()

prompt ="""
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


with langfuse.start_as_current_observation(
    as_type="generation",
    name="ministral-facture",
    model="mistralai/ministral-3-3b",
    input=prompt
) as gen:

    response = client.chat.completions.create(
        model="mistralai/ministral-3-3b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    output = response.choices[0].message.content
    gen.update(output=output, metadata={"backend": "ml-studio-local"})

langfuse.flush()
