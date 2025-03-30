"""Ollama model implementation for GraphRAG."""

import json
import logging
import httpx
from typing import Any, Dict, List, Optional, AsyncGenerator, Generator

from graphrag.language_model.response.base import BaseModelOutput, BaseModelResponse, ModelResponse
from graphrag.language_model.providers.fnllm.utils import _create_cache, _create_error_handler, run_coroutine_sync

log = logging.getLogger(__name__)

class OllamaChatFNLLM:
    """An Ollama Chat Model provider implementation with structured output support."""

    def __init__(
        self,
        *,
        name: str,
        config: Any,  # Should be LanguageModelConfig but for compatibility we use Any
        callbacks: Optional[Any] = None,
        cache: Optional[Any] = None,
    ) -> None:
        """Initialize the OllamaChatFNLLM instance."""
        self.name = name
        self.model = config.model
        # Use either api_base from config or default to localhost
        self.api_base = getattr(config, "api_base", "http://localhost:11434").rstrip("/")
        # Ollama doesn't actually use API keys, so we ignore that setting
        self.api_key = None
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
                         "num_ctx", "repeat_penalty", "num_predict", "stop", 
                         "auth_type", "api_key"]  # Ignore auth settings
        }
        
        # Set up cache if provided
        self.cache = _create_cache(cache, name) if cache else None
        self.error_handler = _create_error_handler(callbacks) if callbacks else None

    def _prepare_messages(self, prompt: str, history: List = None) -> List[Dict[str, str]]:
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
    
    def _prepare_payload(self, messages: List[Dict[str, str]], stream: bool = True) -> Dict[str, Any]:
        """Prepare the payload for the API request."""
        # Only include Ollama-supported options
        supported_options = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "repeat_penalty": self.repeat_penalty,
        }
        
        # Add optional parameters if they're set
        if self.num_predict is not None:
            supported_options["num_predict"] = self.num_predict
        if self.stop is not None:
            supported_options["stop"] = self.stop
            
        # Filter additional parameters to only include Ollama-supported ones
        # Ollama supported parameters include: mirostat, mirostat_eta, mirostat_tau, 
        # num_ctx, num_gpu, num_thread, repeat_last_n, repeat_penalty, temperature, seed, stop, tfs_z
        ollama_supported_params = {
            "mirostat", "mirostat_eta", "mirostat_tau", "num_ctx", "num_gpu", 
            "num_thread", "repeat_last_n", "repeat_penalty", "temperature", 
            "seed", "stop", "tfs_z", "top_k", "top_p"
        }
        
        for param, value in self.additional_params.items():
            if param in ollama_supported_params:
                supported_options[param] = value
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": supported_options,
        }
        
        # Add format for structured output if provided
        if self.format is not None:
            payload["format"] = self.format
            
        return payload

    async def achat(
        self, prompt: str, history: List = None, **kwargs: Any
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
        self, prompt: str, history: List = None, **kwargs: Any
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
        self, prompt: str, history: List = None, **kwargs: Any
    ) -> ModelResponse:
        """Generate a response from the model (sync version)."""
        return run_coroutine_sync(self.achat(prompt, history=history, **kwargs))

    def chat_stream(
        self, prompt: str, history: List = None, **kwargs: Any
    ) -> Generator[str, None, None]:
        """Generate a streaming response from the model (sync version)."""
        msg = "chat_stream is not supported for synchronous execution"
        raise NotImplementedError(msg) 