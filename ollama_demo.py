from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(model='gemma3', messages=[
  {
    'role': 'user',
    'content': 'How many hours a day?',
  },
])
print(response['message']['content'])
# or access fields directly from the response object
# print(response.message.content)
