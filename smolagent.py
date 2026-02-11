from smolagents import CodeAgent, WebSearchTool, InferenceClientModel

model = InferenceClientModel()
agent = CodeAgent(tools=[WebSearchTool()], model=model, stream_outputs=True)

prompt="""
"How many seconds would it take for a leopard at full speed to run through Pont des Arts?"
Alos, what is the weather in Paris.
"""
agent.run(prompt)