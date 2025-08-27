import os
from agents import Agent,OpenAIChatCompletionsModel,AsyncOpenAI, RunConfig, Runner
from dotenv import load_dotenv

#Load environment variables from .env file
load_dotenv()


gemini_api_key = os.getenv("OPENROUTER_API_KEY")
# print("Gemini API Key:", gemini_api_key)    

# Check if the API key is set
if not gemini_api_key:
    raise ValueError("OPENROUTER_API_KEY is not set in the environment variables.") 


# Initialize the OpenAI client with your API key and base URL
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://openrouter.ai/api/v1"  # OpenRouter API base URL  
)

# Model Setup
model=OpenAIChatCompletionsModel(   
    model="deepseek/deepseek-r1-0528:free",#replace model if needed
    openai_client=external_client,
)

# Setup Config
config=RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True   
)

# Agent Setup
agent = Agent(
    name="MathSolver",
    instructions="You are helpful MathSolver.Solve any maths' problems step-by-step.",
) 

# Run the agent with a math problem input
result = Runner.run_sync(
    agent, 
    input="What is 44*4?",
    run_config=config,
)
# Print the final output from the agent
print(result.final_output)
