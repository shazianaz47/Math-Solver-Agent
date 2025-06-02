import os
from agents import Agent,OpenAIChatCompletionsModel, Runner
from openai import AsyncOpenAI
from dotenv import load_dotenv

#Load environment variables from .env file
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL= "https://openrouter.ai/api/v1"

# Initialize the OpenAI client with your API key and base URL
client = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=BASE_URL,
)

# Define the agent with the OpenAI model
agent = Agent(
    name="MathSolver",
    instructions="Solve any math problem step-by-step.",
    model=OpenAIChatCompletionsModel(   
        openai_client=client,
        model="deepseek/deepseek-r1-0528-qwen3-8b:free",
    )
)

result = Runner.run_sync(agent,"What is the sqrt 64?")
print(result.final_output)
