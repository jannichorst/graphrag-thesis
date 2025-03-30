# Using Phi-4 with Ollama in GraphRAG

This document explains how to set up and use Microsoft's Phi-4 model through Ollama in GraphRAG with structured output support.

## Prerequisites

1. Install Ollama: Follow the instructions at [ollama.com](https://ollama.com) to install Ollama for your platform.
2. Pull the Phi-4 model: Run `ollama pull phi4:latest` to download the model.
3. Make sure your GraphRAG installation is up-to-date.

## Configuration

You can configure the Phi-4 model in your settings.yaml file as follows:

```yaml
models:
  phi4_ollama_model:
    type: ollama_chat
    api_base: http://localhost:11434
    model: phi4:latest
    temperature: 0.7
    num_ctx: 8192 
    repeat_penalty: 1.1
    # JSON schema for structured output
    format: 
      schema_type: object
      properties:
        answer:
          type: string
          description: "The answer to the user's question"
        reasoning:
          type: string
          description: "The reasoning process used to arrive at the answer"
        references:
          type: array
          items:
            type: string
          description: "List of references used to generate the answer"
      required: [answer]
    concurrent_requests: 10
    async_mode: threaded
    tokens_per_minute: 50000
    requests_per_minute: 500
```

To use the model for queries, reference this model in your search configuration:

```yaml
local_search:
  chat_model_id: phi4_ollama_model  # Use the Phi-4 model
  embedding_model_id: default_embedding_model
  prompt: "prompts/local_search_system_prompt.txt"
```

## Structured Output

The Ollama integration supports structured output through the `format` parameter, which takes a JSON schema definition. This ensures the model returns responses in a consistent, parseable format.

The schema should follow this structure:

```yaml
format:
  schema_type: object  # The type of schema (usually object)
  properties:          # The properties in your response
    property1:
      type: string     # The type of this property
      description: "Description for the model to understand what to put here"
    property2:
      type: array
      items:
        type: string   # For array properties
      description: "Description for array items"
  required: [property1]  # Which properties are required
```

## Example Usage in Code

Here's how to use the Phi-4 model with structured output in your code:

```python
from graphrag.language_model.factory import ModelFactory
from graphrag.config.enums import ModelType
from graphrag.config.models.language_model_config import LanguageModelConfig

# Create config for Phi-4
config = LanguageModelConfig(
    type=ModelType.OllamaChat,
    model="phi4:latest",
    api_base="http://localhost:11434",
    temperature=0.7,
    format={
        "schema_type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The answer to the user's question"
            }
        },
        "required": ["answer"]
    }
)

# Create the model
model = ModelFactory.create_chat_model(
    model_type=ModelType.OllamaChat,
    name="phi4_example",
    config=config
)

# Use the model
response = await model.achat("What is the capital of France?")

# Access the parsed structured output
if response.parsed_response:
    answer = response.parsed_response.get("answer")
    print(f"Answer: {answer}")
```

## Limitations

- The structured output depends on the model's capabilities. Phi-4 may not always adhere to the schema perfectly.
- Complex schemas with deeply nested structures might be challenging for the model.
- Streaming responses with structured output will stream the JSON chunks, but the complete JSON will only be parsed after the entire response is received.

## Troubleshooting

1. **Model not generating structured output**: Ensure the `format` parameter is correctly defined and the model supports structured output.
2. **Connection errors**: Verify Ollama is running with `ollama list` and check the API base URL.
3. **Poor responses**: Try adjusting temperature, repeat_penalty, and other model parameters to improve output quality.

For more information on Ollama's structured output functionality, see the [official blog post](https://ollama.com/blog/structured-outputs). 