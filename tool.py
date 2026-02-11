from dotenv import load_dotenv
from langfuse import observe, get_client, Evaluation
from groq import Groq
import json
from datetime import datetime

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



@observe(as_type="tool",name="get_weather")
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


@observe(as_type="tool",name="calculate")
def calculate(number1:float,number2:float,operation:str):
    """_summary_

    Args:
        number1 (float): _description_
        number2 (float): _description_
        operation (str): _description_
    """
    try:
        allow=("0123456789./*()")
    except Exception as e:
        print("Erreur:", e)




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

tool_calling_agent()