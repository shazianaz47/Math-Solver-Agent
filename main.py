import os
from agents import Agent, OpenAIChatCompletionsModel,AsyncOpenAI, Runner,set_tracing_disabled
from dotenv import load_dotenv
import rich
#Load environment variables from .env file
load_dotenv()
set_tracing_disabled(disabled=True)

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
# print("Gemini API Key:", gemini_api_key)    
# Check if the API key is set
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY is not set in the environment variables.") 

# Initialize the OpenAI client with your API key and base URL
external_client = AsyncOpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",  # OpenRouter API base URL  
)

# Agent Setup
agent = Agent(
    name="MathSolver",
    instructions="You are helpful MathSolver.Solve any maths' problems step-by-step.",
    model=OpenAIChatCompletionsModel(   
        model="qwen/qwen3-coder:free",#replace model if needed
        openai_client=external_client,
    ),
) 

# Run the agent with a math problem input
result = Runner.run_sync(
    starting_agent=agent, 
    input="Who are you? Solve 12*6+3-4/2"
    # run_config=config,
)
# Print the final output from the agent
rich.print(result.final_output)
