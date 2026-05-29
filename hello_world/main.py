from dotenv import load_dotenv
import os
import anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)

message = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1000,
    messages=[
        {
            "role": "user",
            "content": "What should I search for to find the latest developments in renewable energy?",
        }
    ],
)
print(message)