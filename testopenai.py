import os
import openai

api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = api_key

prompt = "Translate the following English text to French: 'Hello, how are you?'"
response = openai.Completion.create(
    engine="gpt-3.5-turbo-instruct",
    prompt=prompt,
    max_tokens=50
)
print(response.choices[0].text.strip())