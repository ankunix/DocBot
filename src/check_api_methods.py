# Import your DeepSeekAPI implementation
from deepseek import DeepSeekAPI  # Updated import

# Initialize the model with your API key
model = DeepSeekAPI(api_key="sk-1cd746a7018248ada1988f5094d1e629")

# Print available methods
print("Available methods:", [method for method in dir(model) if not method.startswith('_')])

# Let's also try to print docstring to get more info
print("\nDocumentation:")
print(help(model))
