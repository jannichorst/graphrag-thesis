# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing fnllm model provider definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fnllm.openai import (
    create_openai_chat_llm,
    create_openai_client,
    create_openai_embeddings_llm,
)

from graphrag.language_model.providers.fnllm.events import FNLLMEvents
from graphrag.language_model.providers.fnllm.utils import (
    _create_cache,
    _create_error_handler,
    _create_openai_config,
    run_coroutine_sync,
)
from graphrag.language_model.response.base import (
    BaseModelOutput,
    BaseModelResponse,
    ModelResponse,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

    from fnllm.openai.types.client import OpenAIChatLLM as FNLLMChatLLM
    from fnllm.openai.types.client import OpenAIEmbeddingsLLM as FNLLMEmbeddingLLM

    from graphrag.cache.pipeline_cache import PipelineCache
    from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
    from graphrag.config.models.language_model_config import (
        LanguageModelConfig,
    )

# Import additional libraries for Ollama
import json
import logging
import httpx
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

class OpenAIChatFNLLM:
    """An OpenAI Chat Model provider using the fnllm library."""

    model: FNLLMChatLLM

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        model_config = _create_openai_config(config, azure=False)
        error_handler = _create_error_handler(callbacks) if callbacks else None
        model_cache = _create_cache(cache, name)
        client = create_openai_client(model_config)
        self.model = create_openai_chat_llm(
            model_config,
            client=client,
            cache=model_cache,
            events=FNLLMEvents(error_handler) if error_handler else None,
        )

    async def achat(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> ModelResponse:
        """
        Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The response from the Model.
        """
        if history is None:
            response = await self.model(prompt, **kwargs)
        else:
            response = await self.model(prompt, history=history, **kwargs)
        return BaseModelResponse(
            output=BaseModelOutput(content=response.output.content),
            parsed_response=response.parsed_json,
            history=response.history,
            cache_hit=response.cache_hit,
            tool_calls=response.tool_calls,
            metrics=response.metrics,
        )

    async def achat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            A generator that yields strings representing the response.
        """
        if history is None:
            response = await self.model(prompt, stream=True, **kwargs)
        else:
            response = await self.model(prompt, history=history, stream=True, **kwargs)
        async for chunk in response.output.content:
            if chunk is not None:
                yield chunk

    def chat(self, prompt: str, history: list | None = None, **kwargs) -> ModelResponse:
        """
        Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The response from the Model.
        """
        return run_coroutine_sync(self.achat(prompt, history=history, **kwargs))

    def chat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> Generator[str, None]:
        """
        Stream Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            A generator that yields strings representing the response.
        """
        msg = "chat_stream is not supported for synchronous execution"
        raise NotImplementedError(msg)


class OpenAIEmbeddingFNLLM:
    """An OpenAI Embedding Model provider using the fnllm library."""

    model: FNLLMEmbeddingLLM

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        model_config = _create_openai_config(config, azure=False)
        error_handler = _create_error_handler(callbacks) if callbacks else None
        model_cache = _create_cache(cache, name)
        client = create_openai_client(model_config)
        self.model = create_openai_embeddings_llm(
            model_config,
            client=client,
            cache=model_cache,
            events=FNLLMEvents(error_handler) if error_handler else None,
        )

    async def aembed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the LLM.

        Returns
        -------
            The embeddings of the text.
        """
        response = await self.model(text_list, **kwargs)
        if response.output.embeddings is None:
            msg = "No embeddings found in response"
            raise ValueError(msg)
        embeddings: list[list[float]] = response.output.embeddings
        return embeddings

    async def aembed(self, text: str, **kwargs) -> list[float]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        response = await self.model([text], **kwargs)
        if response.output.embeddings is None:
            msg = "No embeddings found in response"
            raise ValueError(msg)
        embeddings: list[float] = response.output.embeddings[0]
        return embeddings

    def embed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the LLM.

        Returns
        -------
            The embeddings of the text.
        """
        return run_coroutine_sync(self.aembed_batch(text_list, **kwargs))

    def embed(self, text: str, **kwargs) -> list[float]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        return run_coroutine_sync(self.aembed(text, **kwargs))


class AzureOpenAIChatFNLLM:
    """An Azure OpenAI Chat LLM provider using the fnllm library."""

    model: FNLLMChatLLM

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        model_config = _create_openai_config(config, azure=True)
        error_handler = _create_error_handler(callbacks) if callbacks else None
        model_cache = _create_cache(cache, name)
        client = create_openai_client(model_config)
        self.model = create_openai_chat_llm(
            model_config,
            client=client,
            cache=model_cache,
            events=FNLLMEvents(error_handler) if error_handler else None,
        )

    async def achat(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> ModelResponse:
        """
        Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            history: The conversation history.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The response from the Model.
        """
        if history is None:
            response = await self.model(prompt, **kwargs)
        else:
            response = await self.model(prompt, history=history, **kwargs)
        return BaseModelResponse(
            output=BaseModelOutput(content=response.output.content),
            parsed_response=response.parsed_json,
            history=response.history,
            cache_hit=response.cache_hit,
            tool_calls=response.tool_calls,
            metrics=response.metrics,
        )

    async def achat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            history: The conversation history.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            A generator that yields strings representing the response.
        """
        if history is None:
            response = await self.model(prompt, stream=True, **kwargs)
        else:
            response = await self.model(prompt, history=history, stream=True, **kwargs)
        async for chunk in response.output.content:
            if chunk is not None:
                yield chunk

    def chat(self, prompt: str, history: list | None = None, **kwargs) -> ModelResponse:
        """
        Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The response from the Model.
        """
        return run_coroutine_sync(self.achat(prompt, history=history, **kwargs))

    def chat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> Generator[str, None]:
        """
        Stream Chat with the Model using the given prompt.

        Args:
            prompt: The prompt to chat with.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            A generator that yields strings representing the response.
        """
        msg = "chat_stream is not supported for synchronous execution"
        raise NotImplementedError(msg)


class AzureOpenAIEmbeddingFNLLM:
    """An Azure OpenAI Embedding Model provider using the fnllm library."""

    model: FNLLMEmbeddingLLM

    def __init__(
        self,
        *,
        name: str,
        config: LanguageModelConfig,
        callbacks: WorkflowCallbacks | None = None,
        cache: PipelineCache | None = None,
    ) -> None:
        model_config = _create_openai_config(config, azure=True)
        error_handler = _create_error_handler(callbacks) if callbacks else None
        model_cache = _create_cache(cache, name)
        client = create_openai_client(model_config)
        self.model = create_openai_embeddings_llm(
            model_config,
            client=client,
            cache=model_cache,
            events=FNLLMEvents(error_handler) if error_handler else None,
        )

    async def aembed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        response = await self.model(text_list, **kwargs)
        if response.output.embeddings is None:
            msg = "No embeddings found in response"
            raise ValueError(msg)
        embeddings: list[list[float]] = response.output.embeddings
        return embeddings

    async def aembed(self, text: str, **kwargs) -> list[float]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        response = await self.model([text], **kwargs)
        if response.output.embeddings is None:
            msg = "No embeddings found in response"
            raise ValueError(msg)
        embeddings: list[float] = response.output.embeddings[0]
        return embeddings

    def embed_batch(self, text_list: list[str], **kwargs) -> list[list[float]]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        return run_coroutine_sync(self.aembed_batch(text_list, **kwargs))

    def embed(self, text: str, **kwargs) -> list[float]:
        """
        Embed the given text using the Model.

        Args:
            text: The text to embed.
            kwargs: Additional arguments to pass to the Model.

        Returns
        -------
            The embeddings of the text.
        """
        return run_coroutine_sync(self.aembed(text, **kwargs))

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
        """Initialize the OllamaChatFNLLM instance.
        
        Parameters
        ----------
        name : str
            The name of the model.
        config : LanguageModelConfig
            The configuration for the model.
        callbacks : WorkflowCallbacks | None, optional
            Callbacks for the model, by default None
        cache : PipelineCache | None, optional
            Cache for the model, by default None
        """
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
        """Format messages for Ollama's chat API.
        
        Parameters
        ----------
        prompt : str
            The prompt to send to the model.
        history : list | None, optional
            Conversation history if available, by default None
            
        Returns
        -------
        list[dict[str, str]]
            A list of message dictionaries in Ollama's format
        """
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
        """Prepare the payload for the API request.
        
        Parameters
        ----------
        messages : list[dict[str, str]]
            The messages to send to the model.
        stream : bool, optional
            Whether to stream the response, by default True
            
        Returns
        -------
        dict[str, Any]
            The prepared payload for the API request.
        """
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
        """Generate a response from the model.
        
        Parameters
        ----------
        prompt : str
            The prompt to send to the model.
        history : list | None, optional
            Conversation history if available, by default None
        **kwargs : Any
            Additional parameters to pass to the model.
            
        Returns
        -------
        ModelResponse
            The model's response.
        """
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
        """Generate a streaming response from the model.
        
        Parameters
        ----------
        prompt : str
            The prompt to send to the model.
        history : list | None, optional
            Conversation history if available, by default None
        **kwargs : Any
            Additional parameters to pass to the model.
            
        Yields
        ------
        str
            Chunks of the generated response.
        """
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
        """Generate a response from the model (sync version).
        
        Parameters
        ----------
        prompt : str
            The prompt to send to the model.
        history : list | None, optional
            Conversation history if available, by default None
        **kwargs : Any
            Additional parameters to pass to the model.
            
        Returns
        -------
        ModelResponse
            The model's response.
        """
        return run_coroutine_sync(self.achat(prompt, history=history, **kwargs))

    def chat_stream(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> Generator[str, None]:
        """Generate a streaming response from the model (sync version).
        
        Parameters
        ----------
        prompt : str
            The prompt to send to the model.
        history : list | None, optional
            Conversation history if available, by default None
        **kwargs : Any
            Additional parameters to pass to the model.
            
        Raises
        ------
        NotImplementedError
            This method is not supported for synchronous execution.
        """
        msg = "chat_stream is not supported for synchronous execution"
        raise NotImplementedError(msg)
