# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Example script demonstrating how to use Ollama with Phi-4 for structured output in GraphRAG."""

import json
import asyncio
from graphrag.language_model.factory import ModelFactory
from graphrag.config.enums import ModelType
from graphrag.config.models.language_model_config import LanguageModelConfig


async def main():
    """Main function to demonstrate Ollama Phi-4 usage with structured output."""
    
    # Define a sample JSON schema for structured output
    schema = {
        "schema_type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The answer to the user's question"
            },
            "reasoning": {
                "type": "string",
                "description": "The reasoning process used to arrive at the answer"
            },
            "references": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "List of references used to generate the answer"
            }
        },
        "required": ["answer"]
    }
    
    # Create a language model config for Ollama with Phi-4
    config = LanguageModelConfig(
        type=ModelType.OllamaChat,
        model="phi4:latest",
        api_base="http://localhost:11434",
        temperature=0.7,
        num_ctx=8192,
        format=schema
    )
    
    # Create the model
    model = ModelFactory.create_chat_model(
        model_type=ModelType.OllamaChat,
        name="phi4_example",
        config=config
    )
    
    # Example prompt that should return structured output
    prompt = """
    Given the following information, please answer the question.
    
    The Phi-4 model is Microsoft's most recent language model in the Phi series, 
    featuring 4 billion parameters and released in July 2024. It was trained on a diverse 
    range of high-quality data and shows strong performance on benchmarks compared to 
    similarly sized models.
    
    Question: When was Phi-4 released and how many parameters does it have?
    """
    
    # Get a response from the model
    print("Sending request to Ollama with Phi-4...")
    response = await model.achat(prompt)
    
    # Display the raw response
    print("\nRaw Response:")
    print(response.output.content)
    
    # Display the parsed JSON if available
    if response.parsed_response:
        print("\nParsed Structured Output:")
        print(json.dumps(response.parsed_response, indent=2))
    else:
        print("\nNo structured output was parsed.")
    
    # Example of streaming (uncomment to try)
    # print("\nStreaming response:")
    # async for chunk in model.achat_stream(prompt):
    #     print(chunk, end="", flush=True)
    # print("\n")


if __name__ == "__main__":
    asyncio.run(main()) 