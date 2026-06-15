"""
Only test compiled models from langchain package
"""

import uuid

from dotenv import load_dotenv
from langchain.agents import create_agent
from langfuse import get_client, observe, propagate_attributes
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

load_dotenv()
AnthropicInstrumentor().instrument()
langfuse_client = get_client()

if langfuse_client.auth_check():
    print("Successfully authenticated with Langfuse.")
else:
    print("Failed to authenticate with Langfuse. Please check your API key and base URL.")


@observe(name="execute_tool")
def check_weather(location: str) -> str:
    """Return the weather forecast for the specified location."""
    return f"It's always sunny in {location}"


@observe(name="run_agent")
def main():
    with propagate_attributes(session_id=str(uuid.uuid4())):
        agent = create_agent(
            model="anthropic:claude-sonnet-4-5-20250929",
            tools=[check_weather],
            system_prompt="You are a weather helper",
        )

        result = agent.invoke({"messages": [{"role": "user", "content": "告诉我上海的天气"}]})
        print(result)


if __name__ == "__main__":
    main()
