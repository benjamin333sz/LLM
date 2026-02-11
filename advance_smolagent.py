from dotenv import load_dotenv
from langfuse import observe, get_client, Evaluation
from groq import Groq
import json
from datetime import datetime
from smolagents import tool,CodeAgent, WebSearchTool, InferenceClientModel
load_dotenv()

groq_client = Groq()



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




# Create agent instances
research_agent = CodeAgent(
    max_steps=5,
    model=model,
    planning_interval=2,
    tools=[WebSearchTool()],
    name="research_agent",
    description="Agent for web research to find relevant tools and agents"
)

analysis_agent = CodeAgent(
    max_steps=5,
    model=model,
    tools=[],
    name="analysis_agent",
    description="Agent for analyzing and summarizing research results"
)

dashboard_agent = CodeAgent(
    max_steps=5,
    model=model,
    tools=[],
    name="dashboard_agent",
    description="Create a Dashboard of the result"
)

def build_multi_agent_system(prompt):
    agent = CodeAgent(
        managed_agents=[research_agent, analysis_agent,dashboard_agent],
        max_steps=8,
        model=model,
        tools=[],
    )
    result = agent.run(prompt)
    return result

sample_prompt = """
Trouve des agents en ligne ou outils pouvant aider à résoudre la demande utilisateur :\n"
"\nExemple: Je veux créer un résumé automatique de documents PDF en français."""

output = build_multi_agent_system(sample_prompt)
print("\nRésultat multi-agent :\n", output)
