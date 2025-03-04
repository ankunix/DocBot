from deepseek import DeepSeekAPI

# Initialize the model
model = DeepSeekAPI(api_key="sk-1cd746a7018248ada1988f5094d1e629")

# Get help on the chat_completion method
print("Documentation for chat_completion:")
print(help(model.chat_completion))

# Try a simple chat completion call
try:
    print("\nTesting chat_completion with minimal parameters:")
    response = model.chat_completion(
        messages=[
            {"role": "user", "content": "Hello, how are you?"}
        ]
    )
    print("Response:", response)
except Exception as e:
    print(f"Error: {str(e)}")

# Try the completion_impl method as an alternative
try:
    print("\nTesting completion_impl as an alternative:")
    response = model.completion_impl(
        prompt="Hello, how are you?"
    )
    print("Response:", response)
except Exception as e:
    print(f"Error: {str(e)}")
