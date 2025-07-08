import os
from openai import AzureOpenAI


sandbox_api_key = os.environ["AI_SANDBOX_KEY"]
sandbox_endpoint = "https://api-ai-sandbox.princeton.edu/"
sandbox_api_version = "2024-02-01"
model_to_be_used = "gpt-4o"

client = AzureOpenAI(
    api_key=sandbox_api_key,
    azure_endpoint = sandbox_endpoint,   
    api_version=sandbox_api_version # current api version not in preview
)

response = client.chat.completions.create(
    model=model_to_be_used,
    temperature=0, # temperature = how creative/random the model is in generating response - 0 to 1 with 1 being most creative
    max_tokens=1000, # max_tokens = token limit on context to send to the model
    top_p=0.01, # top_p = diversity of generated text by the model considering probability attached to token - 0 to 1 - ex. top_p of 0.1 = only tokens within the top 10% probability are considered
    messages=[
        {"role": "system", "content": "You are nobody"}, # describes model identity and purpose
        {"role": "user", "content": "Just say 'Okay'"}, # user prompt
    ]
)
print("Reply:\n" + response.choices[0].message.content)
if response.usage:
    print(f"\nPrompt Tokens Used: {response.usage.prompt_tokens}")
    print(f"Completion Tokens Used: {response.usage.completion_tokens}")
    print(f"Total Tokens Used: {response.usage.total_tokens}")