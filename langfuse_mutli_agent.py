from typing import Any,Dict
from dotenv import load_dotenv
import json
load_dotenv()

# lms server start
# lms server stop
import base64
from openai import OpenAI
from langfuse import get_client,observe

image_path = "invoice/invoice_2.jpeg"
with open(image_path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

langfuse = get_client()

valeur_compte_banque=3000

invoice_planner="""
You are a planning agent.

Your job is to generate an executable plan as JSON.
You must only return JSON. No explanations.

Available actions:
- extract_invoice
- get_bank_balance
- subtract

Goal:
- Extract the total TTC from the invoice image
- Get the current bank balance
- Subtract the invoice amount from the bank balance

Expected JSON format:
{
  "steps": [
    {"id": 1, "action": "extract_invoice"},
    {"id": 2, "action": "get_bank_balance"},
    {"id": 3, "action": "subtract", "input_from": [1,2]}
  ]
}
"""

invoice_extractor="""
Extract invoice information from the image.

Rules:
- Return only valid JSON
- Follow exactly the provided JSON schema
- If a value is missing, return null
- Do not add extra fields
"""

TOOLS_DESCRIPTION = """
Available tools:

- extract_invoice(image_b64: str) -> dict
  Extract structured invoice data from an image.
  Returns a JSON object following the invoice schema.

- get_bank_balance() -> float
  Returns the current bank account balance in EUR.

- subtract_invoice(balance: float, invoice: float) -> float
  Subtracts an invoice amount from the bank balance.
"""

try:
    prompt_plan = langfuse.get_prompt("invoice_planner",label=["production"]).compile()
except Exception:
    langfuse.create_prompt(name="invoice_planner", prompt=invoice_planner, tags=["plan"],labels=["production"])
    prompt_plan = langfuse.get_prompt("invoice_planner",label=["production"]).compile()


try:
    prompt_exec = langfuse.get_prompt("invoice_extractor",label=["production"]).compile()
except Exception:
    langfuse.create_prompt(name="invoice_extractor", prompt=invoice_extractor, tags=["executor"],labels=["production"])
    prompt_exec = langfuse.get_prompt("invoice_extractor",label=["production"]).compile()



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




@observe(name="planification", as_type="agent")
def agent_planification() -> dict:
    response = client.chat.completions.create(
    model="mistralai/ministral-3-3b",
    temperature=0,
    messages=[
            {
                "role": "system",
                "content": langfuse.get_prompt(
                    "invoice_planner",
                    label="latest"
                ).compile()
            }
        ]
    )
    return safe_json_load(response.choices[0].message.content)


@observe(name="execution",as_type="agent")
def agent_executor(plan: dict):
    context = {}

    for step in plan["steps"]:

        if step["action"] == "extract_invoice":
            print("extract")
            context["invoice"] = extract_invoice(image_b64)

        elif step["action"] == "get_bank_balance":
            print("get_bank_balance")
            context["balance"] = get_bank_balance()

        elif step["action"] == "subtract":
            print("subtract")
            context["remaining"] = subtract_invoice(
                context["balance"],
                context["invoice"]["total_ttc"]
            )

    return context


@observe(name="safe_json",as_type="span")
def safe_json_load(output):
    """
    Check if the output is really a json

    Returns :
        json format verification
    """
    start = output.find("{")
    end = output.rfind("}") + 1
    return json.loads(output[start:end])



@observe(name="get_bank_balance", as_type="tool")
def get_bank_balance() -> float:
    """
    Return the value in the bank account on e

    Returns :
        Value in the bank account on €
    """
    return valeur_compte_banque

@observe(name="subtract_bank_value",as_type="tool")
def subtract_invoice(balance: float, invoice: float) -> float:
    """
    Substract the value of the bank account and an invoice.

    Returns :
        Value in the bank account on € after the payement
    """
    return balance - invoice

@observe(name="extract_invoice", as_type="tool")
def extract_invoice(image_b64: str) -> dict:
    response = client.chat.completions.create(
        model="mistralai/ministral-3-3b",
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": schema
        },
        messages=[
            {
                "role": "system",
                "content": langfuse.get_prompt("invoice_extractor").compile()
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Voici la facture."},
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
    raw_output = response.choices[0].message.content
    return safe_json_load(raw_output)


with langfuse.start_as_current_observation(
    as_type="generation",
    name="multi_agent_facture",
    model="mistralai/ministral-3-3b",
    input="invoice processing"
):
    plan = agent_planification()
    print("début execution")
    result = agent_executor(plan)


langfuse.flush()
