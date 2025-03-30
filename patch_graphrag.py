"""Patch the GraphRAG codebase to add Ollama support.

This script copies the Ollama implementation files into the appropriate
GraphRAG directories to add Ollama support to the framework.
"""

import os
import sys
import shutil
import importlib.util
import subprocess
from pathlib import Path

def get_graphrag_directory():
    """Find the GraphRAG installation directory."""
    try:
        # Try to import graphrag to find its location
        graphrag_spec = importlib.util.find_spec("graphrag")
        if graphrag_spec is None:
            print("GraphRAG package not found.")
            return None
        
        graphrag_dir = Path(graphrag_spec.origin).parent
        return graphrag_dir
    except ImportError:
        print("GraphRAG package not found.")
        return None

def install_dependencies():
    """Install required dependencies for Ollama integration."""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx"])
        print("Successfully installed dependencies.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False
    return True

def create_ollama_implementation(graphrag_dir):
    """Create the Ollama implementation files in the GraphRAG codebase."""
    # Ensure the model provider directory exists
    fnllm_dir = graphrag_dir / "language_model" / "providers" / "fnllm"
    if not fnllm_dir.exists():
        print(f"FNLLM provider directory not found at {fnllm_dir}")
        return False
    
    # Check if OllamaChatFNLLM is already in models.py
    models_file = fnllm_dir / "models.py"
    if models_file.exists():
        with open(models_file, 'r') as f:
            content = f.read()
            if "OllamaChatFNLLM" in content:
                print("Ollama implementation already exists in models.py")
                return True
    
    # Create Ollama implementation
    print("Creating Ollama implementation...")
    
    # Create Ollama implementation code
    ollama_code = '''
# Add OllamaChatFNLLM class
class OllamaChatFNLLM:
    """An Ollama Chat Model provider implementation with structured output support."""

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        """Initialize the OllamaChatFNLLM instance."""
        self.name = name
        self.model = config.model
        # Use either api_base from config or default to localhost
        self.api_base = getattr(config, "api_base", "http://localhost:11434").rstrip("/")
        # Get JSON schema format if provided
        self.format = getattr(config, "format", None)
        # Other Ollama parameters
        self.temperature = getattr(config, "temperature", 0.7)
        self.num_ctx = getattr(config, "num_ctx", 4096)
        self.repeat_penalty = getattr(config, "repeat_penalty", 1.1)
        self.num_predict = getattr(config, "num_predict", None)
        self.stop = getattr(config, "stop", None)
        # Store any other parameters from config
        self.additional_params = {
            k: v for k, v in config.__dict__.items() 
            if k not in ["model", "api_base", "format", "temperature", 
                         "num_ctx", "repeat_penalty", "num_predict", "stop"]
        }
        
        # Set up cache if provided
        self.cache = _create_cache(cache, name) if cache else None
        self.error_handler = _create_error_handler(callbacks) if callbacks else None

    def _prepare_messages(self, prompt: str, history: list | None = None) -> list[dict[str, str]]:
        """Format messages for Ollama's chat API."""
        messages = []
        
        # Add history messages if provided
        if history:
            for message in history:
                if isinstance(message, dict):
                    # Handle message dictionaries with role/content format
                    role = message.get("role", "user")
                    content = message.get("content", "")
                    messages.append({"role": role, "content": content})
                elif isinstance(message, str):
                    # Default to user role for string messages
                    messages.append({"role": "user", "content": message})
        
        # Add the current prompt as a user message
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    def _prepare_payload(self, messages: list[dict[str, str]], stream: bool = True) -> dict[str, Any]:
        """Prepare the payload for the API request."""
        options = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "repeat_penalty": self.repeat_penalty,
        }
        
        # Add optional parameters if they're set
        if self.num_predict is not None:
            options["num_predict"] = self.num_predict
        if self.stop is not None:
            options["stop"] = self.stop
            
        # Add any additional parameters
        options.update(self.additional_params)
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": options,
        }
        
        # Add format for structured output if provided
        if self.format is not None:
            payload["format"] = self.format
            
        return payload

    async def achat(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> ModelResponse:
        """Generate a response from the model."""
        messages = self._prepare_messages(prompt, history)
        payload = self._prepare_payload(messages, stream=False)
        
        # Add any additional kwargs to options
        if kwargs:
            payload["options"].update(kwargs)
        
        async with httpx.AsyncClient(base_url=self.api_base, timeout=None) as client:
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
            
            response_json = resp.json()
            content = response_json.get("message", {}).get("content", "")
            
            # Check if the response has a parsed JSON structure
            parsed_json = None
            if self.format is not None:
                try:
                    parsed_json = json.loads(content)
                except json.JSONDecodeError:
                    log.warning("Failed to parse JSON response despite using format option")
            
            # Create response with message history
            chat_history = messages + [{"role": "assistant", "content": content}]
            
            return BaseModelResponse(
                output=BaseModelOutput(content=content),
                parsed_response=parsed_json,
                history=chat_history,
                cache_hit=False,
                tool_calls=[],
                metrics=None,
            )

    async def achat_stream(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response from the model."""
        messages = self._prepare_messages(prompt, history)
        payload = self._prepare_payload(messages, stream=True)
        
        # Add any additional kwargs to options
        if kwargs:
            payload["options"].update(kwargs)
            
        async with httpx.AsyncClient(base_url=self.api_base, timeout=None) as client:
            async with client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    if chunk.get("done", False):
                        # End of stream
                        break
                    
                    if "message" in chunk:
                        content = chunk["message"].get("content", "")
                        if content:
                            yield content

    def chat(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> ModelResponse:
        """Generate a response from the model (sync version)."""
        return run_coroutine_sync(self.achat(prompt, history=history, **kwargs))

    def chat_stream(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> Generator[str, None]:
        """Generate a streaming response from the model (sync version)."""
        msg = "chat_stream is not supported for synchronous execution"
        raise NotImplementedError(msg)
'''
    
    # Add import for httpx to models.py
    with open(models_file, 'r') as f:
        content = f.read()
    
    # Check if httpx import is already present
    if "import httpx" not in content:
        import_section_end = content.find("from graphrag.")
        if import_section_end > 0:
            modified_content = content[:import_section_end] + "# Import additional libraries for Ollama\nimport json\nimport logging\nimport httpx\nfrom typing import Any, Dict, List, Optional\n\n" + content[import_section_end:]
            
            # Add a logger at the top of the file if needed
            if "log = logging.getLogger(__name__)" not in modified_content:
                log_section_end = modified_content.find("class ")
                if log_section_end > 0:
                    modified_content = modified_content[:log_section_end] + "log = logging.getLogger(__name__)\n\n" + modified_content[log_section_end:]
        else:
            print("Could not find import section in models.py")
            return False
    else:
        modified_content = content
    
    # Append Ollama implementation
    if "class OllamaChatFNLLM" not in modified_content:
        modified_content += ollama_code
    
    # Write modified content back to models.py
    with open(models_file, 'w') as f:
        f.write(modified_content)
    
    # Update the factory.py to register the Ollama chat model
    factory_file = graphrag_dir / "language_model" / "factory.py"
    if factory_file.exists():
        with open(factory_file, 'r') as f:
            content = f.read()
        
        # Update imports if needed
        if "OllamaChatFNLLM" not in content:
            import_section = content.find("from graphrag.language_model.providers.fnllm.models import (")
            if import_section > 0:
                end_of_imports = content.find(")", import_section)
                if end_of_imports > 0:
                    modified_content = content[:end_of_imports] + ",\n    OllamaChatFNLLM" + content[end_of_imports:]
                else:
                    print("Could not find end of import section in factory.py")
                    return False
            else:
                print("Could not find import section in factory.py")
                return False
        else:
            modified_content = content
        
        # Add registration for Ollama chat model if needed
        if "ModelType.OllamaChat" not in modified_content:
            register_section = modified_content.find("# --- Register default implementations ---")
            if register_section > 0:
                openai_reg_end = modified_content.find(")", modified_content.find("ModelType.OpenAIChat") + 1)
                if openai_reg_end > 0:
                    registration_code = '''\nModelFactory.register_chat(
    ModelType.OllamaChat, lambda **kwargs: OllamaChatFNLLM(**kwargs)
)'''
                    modified_content = modified_content[:openai_reg_end+1] + registration_code + modified_content[openai_reg_end+1:]
                else:
                    print("Could not find OpenAI registration in factory.py")
                    return False
            else:
                print("Could not find registration section in factory.py")
                return False
        
        # Write modified content back to factory.py
        with open(factory_file, 'w') as f:
            f.write(modified_content)
    else:
        print(f"Factory file not found at {factory_file}")
        return False
    
    print("Successfully added Ollama implementation to GraphRAG.")
    return True

def main():
    """Main function to patch GraphRAG with Ollama support."""
    print("Patching GraphRAG with Ollama support...")
    
    # Install dependencies
    if not install_dependencies():
        print("Failed to install dependencies. Exiting.")
        return False
    
    # Find GraphRAG directory
    graphrag_dir = get_graphrag_directory()
    if not graphrag_dir:
        print("Could not find GraphRAG installation. Exiting.")
        return False
    
    print(f"Found GraphRAG at: {graphrag_dir}")
    
    # Create Ollama implementation
    if not create_ollama_implementation(graphrag_dir):
        print("Failed to create Ollama implementation. Exiting.")
        return False
    
    print("Successfully patched GraphRAG with Ollama support!")
    print("You can now use the 'ollama_chat' model type in your settings.yaml file.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 