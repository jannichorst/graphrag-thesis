# GraphRAG

👉 [Use the GraphRAG Accelerator solution](https://github.com/Azure-Samples/graphrag-accelerator) <br/>
👉 [Microsoft Research Blog Post](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)<br/>
👉 [Read the docs](https://microsoft.github.io/graphrag)<br/>
👉 [GraphRAG Arxiv](https://arxiv.org/pdf/2404.16130)

<div align="left">
  <a href="https://pypi.org/project/graphrag/">
    <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/graphrag">
  </a>
  <a href="https://pypi.org/project/graphrag/">
    <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/graphrag">
  </a>
  <a href="https://github.com/microsoft/graphrag/issues">
    <img alt="GitHub Issues" src="https://img.shields.io/github/issues/microsoft/graphrag">
  </a>
  <a href="https://github.com/microsoft/graphrag/discussions">
    <img alt="GitHub Discussions" src="https://img.shields.io/github/discussions/microsoft/graphrag">
  </a>
</div>

## Overview

The GraphRAG project is a data pipeline and transformation suite that is designed to extract meaningful, structured data from unstructured text using the power of LLMs.

To learn more about GraphRAG and how it can be used to enhance your LLM's ability to reason about your private data, please visit the <a href="https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/" target="_blank">Microsoft Research Blog Post.</a>

## Quickstart

To get started with the GraphRAG system we recommend trying the [Solution Accelerator](https://github.com/Azure-Samples/graphrag-accelerator) package. This provides a user-friendly end-to-end experience with Azure resources.

## Repository Guidance

This repository presents a methodology for using knowledge graph memory structures to enhance LLM outputs. Please note that the provided code serves as a demonstration and is not an officially supported Microsoft offering.

⚠️ *Warning: GraphRAG indexing can be an expensive operation, please read all of the documentation to understand the process and costs involved, and start small.*

## Diving Deeper

- To learn about our contribution guidelines, see [CONTRIBUTING.md](./CONTRIBUTING.md)
- To start developing _GraphRAG_, see [DEVELOPING.md](./DEVELOPING.md)
- Join the conversation and provide feedback in the [GitHub Discussions tab!](https://github.com/microsoft/graphrag/discussions)

## Prompt Tuning

Using _GraphRAG_ with your data out of the box may not yield the best possible results.
We strongly recommend to fine-tune your prompts following the [Prompt Tuning Guide](https://microsoft.github.io/graphrag/prompt_tuning/overview/) in our documentation.

## Versioning

Please see the [breaking changes](./breaking-changes.md) document for notes on our approach to versioning the project.

*Always run `graphrag init --root [path] --force` between minor version bumps to ensure you have the latest config format. Run the provided migration notebook between major version bumps if you want to avoid re-indexing prior datasets. Note that this will overwrite your configuration and prompts, so backup if necessary.*

## Responsible AI FAQ

See [RAI_TRANSPARENCY.md](./RAI_TRANSPARENCY.md)

- [What is GraphRAG?](./RAI_TRANSPARENCY.md#what-is-graphrag)
- [What can GraphRAG do?](./RAI_TRANSPARENCY.md#what-can-graphrag-do)
- [What are GraphRAG's intended use(s)?](./RAI_TRANSPARENCY.md#what-are-graphrags-intended-uses)
- [How was GraphRAG evaluated? What metrics are used to measure performance?](./RAI_TRANSPARENCY.md#how-was-graphrag-evaluated-what-metrics-are-used-to-measure-performance)
- [What are the limitations of GraphRAG? How can users minimize the impact of GraphRAG's limitations when using the system?](./RAI_TRANSPARENCY.md#what-are-the-limitations-of-graphrag-how-can-users-minimize-the-impact-of-graphrags-limitations-when-using-the-system)
- [What operational factors and settings allow for effective and responsible use of GraphRAG?](./RAI_TRANSPARENCY.md#what-operational-factors-and-settings-allow-for-effective-and-responsible-use-of-graphrag)

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## Privacy

[Microsoft Privacy Statement](https://privacy.microsoft.com/en-us/privacystatement)

# Ollama Integration for GraphRAG

This project adds support for using [Ollama](https://ollama.ai/) models with the GraphRAG framework, with a specific focus on enabling structured output from local models.

## Prerequisites

- GraphRAG installed in your environment
- Ollama installed and running on your system
- Python 3.9+ with required dependencies

## Installation

1. Make sure you have the required dependencies:
   ```bash
   pip install httpx
   ```

2. Clone this repository or download the files into your project directory:
   ```bash
   # If using git
   git clone https://github.com/yourusername/graphrag-ollama.git
   # Or just download the files directly
   ```

3. Pull your desired Ollama model:
   ```bash
   ollama pull phi4:latest
   # or another model of your choice
   ```

## Using the Ollama Integration

1. Start by registering the Ollama model with GraphRAG:
   ```python
   from graphrag_patch.register import register_ollama
   
   # Register the Ollama model type
   register_ollama()
   ```

2. Configure your model in settings.yaml:
   ```yaml
   language_models:
     default:
       type: openai_chat
       # ... existing settings ...
     
     ollama:
       type: ollama_chat
       name: phi4:latest
       api_base: http://localhost:11434
       temperature: 0.7
       num_ctx: 4096
       json: true
       json_strategy: valid
   
   # In your workflow sections, reference the ollama model:
   workflows:
     index:
       graph_intelligence:
         default:
           language_model_id: ollama
           # ... other settings ...
   ```

3. Alternatively, you can configure the model programmatically:
   ```python
   from graphrag.config.models.language_model_config import LanguageModelConfig
   from graphrag.language_model.providers.fnllm.fnllm import FNLLM
   from graphrag.config.enums import ModelType, JsonStrategy
   
   # Define a schema for structured output
   json_schema = {
       "type": "object",
       "properties": {
           "answer": {"type": "string"},
           "reasoning": {"type": "string"}
       },
       "required": ["answer"]
   }
   
   # Configure the model
   config = LanguageModelConfig(
       type=ModelType.OllamaChat,
       name="phi4:latest",
       api_base="http://localhost:11434",
       temperature=0.7,
       num_ctx=4096,
       json=True,
       json_schema=json_schema,
       json_strategy=JsonStrategy.VALID,
   )
   
   # Create the model instance
   llm = FNLLM(config=config)
   
   # Use the model
   response = await llm.achat(prompt="Your prompt here")
   ```

## Example Usage

Check out the `use_ollama.py` file for a complete example of how to use the Ollama integration with GraphRAG.

## Models Tested

- phi4:latest - Works well with structured output
- llama3:latest - Works well with larger context windows

## Troubleshooting

If you encounter issues:

1. Ensure Ollama is running:
   ```bash
   ollama serve
   ```

2. Verify your model is pulled:
   ```bash
   ollama list
   ```

3. Check that you're using the correct model name and API endpoint in your configuration.

4. For structured output issues, try using the LOOSE json_strategy or simplify your JSON schema.

## How It Works

This integration:

1. Implements the `OllamaChatFNLLM` class that interfaces with Ollama's API
2. Registers the model type with GraphRAG's ModelFactory
3. Handles message formatting and JSON parsing for structured output

5. If you get an error about "Model type ollama_chat is not recognized", you need to run the activation script again.

## Ollama-supported Parameters

When configuring your Ollama model in settings.yaml, be aware that Ollama only supports a specific set of parameters:

- `temperature`: Controls randomness (0.0 to 1.0, higher = more random)
- `num_ctx`: Maximum context size in tokens
- `repeat_penalty`: Penalty for repeating tokens
- `top_p`: Nucleus sampling (0.0 to 1.0)
- `top_k`: Limits vocabulary to top K options
- `num_gpu`: Number of GPUs to use
- `num_thread`: Number of CPU threads to use
- `seed`: Random seed for reproducibility
- `stop`: Stop sequences (when to stop generation)
- `mirostat`, `mirostat_eta`, `mirostat_tau`: Mirostat sampling parameters 
- `repeat_last_n`: Consider last N tokens for penalties
- `tfs_z`: TFS-Z sampling parameter

Other GraphRAG parameters like `concurrent_requests`, `async_mode`, `tokens_per_minute`, etc. are still required in your configuration for GraphRAG's internal handling but won't be passed to Ollama.

Example configuration:
```yaml
models:
  phi4_ollama_model:
    type: ollama_chat
    api_base: http://localhost:11434
    model: phi4:latest
    auth_type: api_key
    api_key: dummy-api-key  # Required by GraphRAG validation but not used by Ollama
    encoding_model: cl100k_base  # Required for GraphRAG's tokenization
    # Ollama-specific parameters:
    temperature: 0.7
    num_ctx: 8192 
    repeat_penalty: 1.1
    # Other supported parameters:
    # top_p: 0.9
    # top_k: 40
    # Other GraphRAG-required parameters (not passed to Ollama)
    concurrent_requests: 10
    async_mode: threaded
    tokens_per_minute: 50000
    requests_per_minute: 500
```

# Using with GraphRAG CLI

Our examples so far have shown how to use Ollama programmatically, but to use it with GraphRAG's command line tools, you need to activate the integration first:

## Activating Ollama for GraphRAG CLI

1. Run the activation script:
   ```bash
   python activate_ollama_cli.py
   ```

   This will:
   - Install the graphrag_patch package in your Python environment
   - Create a hook to automatically load the integration when GraphRAG runs
   - Verify that the Ollama model is properly registered

2. Once activated, you can use GraphRAG CLI commands with Ollama models:
   ```bash
   graphrag index --root ./your_project_dir
   graphrag query --root ./your_project_dir --search-method local --query "Your query here"
   ```

3. Make sure your settings.yaml includes the Ollama model configuration:
   ```yaml
   models:
     phi4_ollama_model:
       type: ollama_chat
       api_base: http://localhost:11434
       model: phi4:latest
       auth_type: api_key
       api_key: dummy-api-key  # Required by GraphRAG validation but not used by Ollama
       encoding_model: cl100k_base
       temperature: 0.7
       num_ctx: 8192
       # Other settings...
   ```

## Troubleshooting CLI Integration

If you encounter errors with the GraphRAG CLI:

1. Make sure Ollama is running:
   ```bash
   ollama serve
   ```

2. Check that your model is available:
   ```bash
   ollama list
   ```

3. Verify that the activation script completed successfully.

4. Try running one of the example scripts directly to test the integration:
   ```bash
   python use_ollama.py
   ```

5. If you get an error about "Model type ollama_chat is not recognized", you need to run the activation script again.
