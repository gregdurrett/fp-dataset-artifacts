# Making requests in Python

1. Install openai dependency
```sh
pip install openai
```

2. Make a request
```py
import os
import openai

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(engine="davinci", prompt="This is a test", max_tokens=5)
```