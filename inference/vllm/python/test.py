# Compatible with OpenAI Python library v1.0.0 and above

from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:11434/v1/",
    api_key="custom"
)

messages = [
    {"role": "user", "content": "hello!"},
    {"role": "user", "content": "Say this is test?"}
]

response = client.chat.completions.create(
    model="llama3.2:latest",
    messages=messages
)

content = response.choices[0].message.content

print(content)
