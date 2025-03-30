"""Patching module for GraphRAG to add Ollama support."""

__version__ = "0.1.0"

# Automatically register Ollama model when importing this package
try:
    from graphrag_patch.register import register_ollama
    register_ollama()
    print("GraphRAG-Ollama integration initialized.")
except Exception as e:
    print(f"Warning: Unable to register Ollama model: {e}") 