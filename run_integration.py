#!/usr/bin/env python3
"""
Helper script to set up and run the Ollama integration for GraphRAG.

This script:
1. Checks and installs dependencies
2. Verifies that Ollama is running
3. Registers the Ollama model with GraphRAG
4. Updates settings.yaml with Ollama configuration (optional)
5. Runs a test query to verify everything is working
"""

import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path

# Try to import required modules
def check_and_install_dependencies():
    """Check and install required dependencies."""
    required_packages = ["httpx"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Installing missing packages: {', '.join(missing_packages)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("Installation complete.")
    else:
        print("All required dependencies are already installed.")

def check_ollama_running():
    """Check if Ollama server is running."""
    import httpx
    
    try:
        response = httpx.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            version = response.json().get("version", "unknown")
            print(f"Ollama is running (version: {version})")
            return True
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("\nPlease start Ollama by running 'ollama serve' in a separate terminal.")
        return False

def check_model_pulled(model_name="phi4:latest"):
    """Check if the specified model is pulled in Ollama."""
    import httpx
    
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            for model in models:
                if model.get("name") == model_name:
                    print(f"Model {model_name} is available.")
                    return True
            
            print(f"Model {model_name} not found. Please pull it by running:")
            print(f"  ollama pull {model_name}")
            return False
    except Exception as e:
        print(f"Error checking models: {e}")
        return False

async def test_model():
    """Run a test query to verify the integration is working."""
    try:
        # Try to import from the local package first
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from graphrag_patch.register import register_ollama
        from graphrag.config.models.language_model_config import LanguageModelConfig
        from graphrag.language_model.factory import ModelFactory
        from graphrag.config.enums import ModelType, AuthType
        # Import JsonStrategy from the correct location
        from fnllm.base.config import JsonStrategy
        
        # Register the model
        register_success = register_ollama()
        if not register_success:
            print("Failed to register Ollama model type.")
            return False
        
        # Simple schema for testing
        test_schema = {
            "schema_type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "description": "The response to the question"
                }
            },
            "required": ["response"]
        }
        
        # Configure the model
        config = LanguageModelConfig(
            type=ModelType.OllamaChat,
            model="phi4:latest",
            name="phi4:latest",
            api_base="http://localhost:11434",
            auth_type=AuthType.APIKey,  # Use the proper enum value
            api_key="dummy-api-key",  # Use a dummy API key to satisfy validation
            temperature=0.7,
            num_ctx=4096,
            encoding_model="cl100k_base",  # Specify tokenizer explicitly
            json=True,
            format=test_schema,  # Use format instead of json_schema for Ollama
            json_strategy=JsonStrategy.VALID,
        )
        
        # Create an LLM instance using the ModelFactory
        llm = ModelFactory.create_chat_model(
            model_type=ModelType.OllamaChat,
            name="phi4:latest", 
            config=config
        )
        
        # Simple test prompt
        test_prompt = "Please respond with a short greeting."
        
        print("Sending test query to Ollama model...")
        response = await llm.achat(prompt=test_prompt)
        
        print("\nReceived response from model:")
        print("---")
        print(response.output.content)
        print("---")
        
        if response.parsed_response:
            print("\nStructured output:")
            print(json.dumps(response.parsed_response, indent=2))
        
        return True
    
    except Exception as e:
        print(f"Error testing the model: {e}")
        return False

def main():
    """Main function to set up and test the Ollama integration."""
    print("Setting up Ollama integration for GraphRAG...\n")
    
    # Step 1: Check dependencies
    print("Step 1: Checking dependencies...")
    check_and_install_dependencies()
    print()
    
    # Step 2: Check if Ollama is running
    print("Step 2: Checking if Ollama is running...")
    if not check_ollama_running():
        print("Please start Ollama and try again.")
        return
    print()
    
    # Step 3: Check if model is pulled
    print("Step 3: Checking if phi4 model is available...")
    if not check_model_pulled():
        print("Please pull the model and try again.")
        return
    print()
    
    # Step 4: Test the model
    print("Step 4: Testing the integration...")
    success = asyncio.run(test_model())
    print()
    
    if success:
        print("✅ Ollama integration is working correctly!")
        print("\nNext steps:")
        print("1. Configure your settings.yaml file to use the Ollama model")
        print("2. Check the README.md file for detailed usage instructions")
        print("3. Try running a full GraphRAG workflow with Ollama")
    else:
        print("❌ There was an issue setting up the Ollama integration.")
        print("Please check the error messages above and try again.")

if __name__ == "__main__":
    main() 