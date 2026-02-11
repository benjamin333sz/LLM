from dotenv import load_dotenv
from langfuse import observe, get_client, Evaluation
from groq import Groq
import json
from datetime import datetime
from smolagents import tool,CodeAgent, WebSearchTool, InferenceClientModel
load_dotenv()

groq_client = Groq()

tools=[
    {
        "type":"function",
        "function":{
            "name":"get_weather",
            "description":"Get the current weather for a given city.",
            "paremeters":{
                "type":"object",
                "properties":{
                    "city":{
                        "type":"string",
                        "description":"The city name, e.g. 'Paris'"
                    }
                },
                "required":["city"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"calculate",
            "description":"Do mathematic operation",
            "paremeters":{
                "type":"object",
                "properties":{
                    "number1":{
                        "type":"float",
                        "description":"First number"
                    },
                    "number2":{
                        "type":"float",
                        "description":"Second number"
                    },
                    "operation":{
                        "type":"string",
                        "description":"operation"
                    }

                },
                "required":["number1","number2","operation"]
            }
        }
    }
]

model = InferenceClientModel(model_id="openai/gpt-oss-120b")

@tool
def get_weather(city:str)->str:
    """
    Get the weather in a specific city

    Args:
        city (str): name of the city

    Returns:
        Dict: information of the weather of the city
    """
    fake_data={
        "Paris":"15°C, cloud",
        "London":"12°C,rainy",
        "Tokyo":"22°C, sunny"
    }
    
    return fake_data.get(city,f"No Weather data available for {city}")



@tool
def calculate(number1:float,number2:float,operation:str)->float:
    """Perform a mathematical operation on two numbers

    Args:
        number1 (float): First number
        number2 (float): Second number
        operation (str): Operation to perform (+, -, *, /)

    Returns:
        float: Result of the operation
    """
    try:
        if operation == "+":
            return float(number1 + number2)
        elif operation == "-":
            return float(number1 - number2)
        elif operation == "*":
            return float(number1 * number2)
        elif operation == "/":
            if number2 == 0:
                return float(0)
            return float(number1 / number2)
        else:
            return float(0)
    except Exception as e:
        print("Erreur:", e)
        return float(0)




def tool_calling_agent(user_message:str)->str:
    response=groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": ""
            },
            {"role": "user", "content": user_message}
        ],
        temperature=0.5,
        tools=tools
    )

@observe()
def run_code_agent(prompt:str):
    agent=CodeAgent(
        tools=[get_weather,calculate],
        model=model,
        max_steps=5
    )
    result=agent.run(prompt)
    return result
prompt="""
Donne moi le temps à Paris, Tokyo et Londre et dit moi laquelle des villes a la plus basse température.
"""
run_code_agent(prompt)