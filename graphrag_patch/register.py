"""Registration module for Ollama model support in GraphRAG."""

import sys
import subprocess
from graphrag.config.enums import ModelType
from graphrag.language_model.factory import ModelFactory
from graphrag_patch.models import OllamaChatFNLLM

def ensure_dependencies():
    """Ensure required dependencies are installed."""
    try:
        import httpx
    except ImportError:
        print("Installing httpx...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx"])
        print("Successfully installed httpx.")

def register_ollama():
    """Register the Ollama model type with GraphRAG."""
    ensure_dependencies()
    
    print("Registering Ollama model with GraphRAG...")
    ModelFactory.register_chat(
        ModelType.OllamaChat, lambda **kwargs: OllamaChatFNLLM(**kwargs)
    )
    
    if ModelFactory.is_supported_chat_model(ModelType.OllamaChat):
        print(f"Successfully registered {ModelType.OllamaChat} model type.")
        return True
    else:
        print(f"Failed to register {ModelType.OllamaChat} model type.")
        return False

if __name__ == "__main__":
    success = register_ollama()
    print("Available chat models:", ModelFactory.get_chat_models())
    sys.exit(0 if success else 1) 