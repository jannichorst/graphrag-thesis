"""
A simple script to demonstrate using Ollama models with GraphRAG.

This script shows how to:
1. Register the Ollama model type with GraphRAG
2. Configure and use the Ollama model for a simple query
3. Handle structured JSON output from the model

Before running this script:
- Make sure the Ollama server is running
- Ensure the phi4 model is pulled (run `ollama pull phi4:latest`)
"""

import asyncio
import json
from graphrag_patch.register import register_ollama
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.factory import ModelFactory
from graphrag.config.enums import ModelType, AuthType
from fnllm.base.config import JsonStrategy

async def main():
    # Register Ollama with GraphRAG
    register_ollama()
    
    # Define JSON schema for structured output
    example_schema = {
        "schema_type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The main answer to the question"
            },
            "reasoning": {
                "type": "string",
                "description": "The reasoning process used to arrive at the answer"
            },
            "confidence": {
                "type": "integer",
                "description": "Confidence level from 1-10"
            }
        },
        "required": ["answer", "reasoning", "confidence"]
    }
    
    # Configure the Ollama model
    config = LanguageModelConfig(
        type=ModelType.OllamaChat,
        model="phi4:latest",  # Required model field
        name="phi4:latest",  # Make sure you've pulled this model: ollama pull phi4:latest
        api_base="http://localhost:11434",  # Default Ollama API endpoint
        auth_type=AuthType.APIKey,  # Use the proper enum value
        api_key="dummy-api-key",  # Use a dummy API key to satisfy validation
        temperature=0.7,
        num_ctx=4096,  # Adjust based on your model's capabilities
        encoding_model="cl100k_base",  # Specify tokenizer explicitly
        json=True,  # Enable JSON mode
        format=example_schema,  # Use format instead of json_schema for Ollama
        json_strategy=JsonStrategy.VALID,  # Strict JSON validation
    )
    
    # Create an LLM instance using the ModelFactory
    llm = ModelFactory.create_chat_model(
        model_type=ModelType.OllamaChat,
        name="phi4:latest", 
        config=config
    )
    
    # Prepare a test prompt
    prompt = """
    What is GraphRAG and what is it used for? 
    Provide a structured response with your answer, reasoning, and confidence level.
    """
    
    print("Sending query to Ollama model...")
    try:
        # Send the prompt to the model
        response = await llm.achat(prompt=prompt)
        
        # Print the response
        print("\nRaw response:")
        print(response.output.content)
        
        # Print structured output if available
        if response.parsed_response is not None:
            print("\nStructured output:")
            print(json.dumps(response.parsed_response, indent=2))
        
        # Print metrics if available
        if response.metrics:
            print("\nMetrics:")
            for key, value in response.metrics.items():
                print(f"{key}: {value}")
    
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure the Ollama server is running")
        print("2. Verify you've pulled the phi4 model (run 'ollama pull phi4:latest')")
        print("3. Check that the API endpoint is correct")

if __name__ == "__main__":
    asyncio.run(main()) 