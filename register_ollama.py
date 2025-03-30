"""Register the Ollama model with GraphRAG's ModelFactory.

This script ensures the OllamaChatFNLLM is properly registered with the ModelFactory.
"""

import sys
import os

# Add the project root to the Python path if needed
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from graphrag.config.enums import ModelType
from graphrag.language_model.factory import ModelFactory
from graphrag.language_model.providers.fnllm.models import OllamaChatFNLLM

# Register the Ollama model with the ModelFactory
print("Registering OllamaChatFNLLM with ModelFactory...")
ModelFactory.register_chat(
    ModelType.OllamaChat, lambda **kwargs: OllamaChatFNLLM(**kwargs)
)

# Verify registration
if ModelFactory.is_supported_chat_model(ModelType.OllamaChat):
    print(f"Successfully registered {ModelType.OllamaChat} model type.")
else:
    print(f"Failed to register {ModelType.OllamaChat} model type.")

print("Available chat models:", ModelFactory.get_chat_models()) 