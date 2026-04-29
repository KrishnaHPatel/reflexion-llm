"""
LLM API Wrapper

This module provides a simple interface to call language models.
Supports multiple providers including free options:
  - Ollama (free, local)
  - Groq (free tier)
  - OpenAI (paid)
"""

import os
from openai import OpenAI


def get_client(provider: str = "ollama") -> OpenAI:
    """
    Create an OpenAI-compatible client for the specified provider.
    
    Args:
        provider: One of "ollama", "groq", or "openai"
    
    Returns:
        OpenAI client configured for the provider
    """
    if provider == "ollama":
        # Ollama runs locally - completely free
        # Install: https://ollama.ai then run: ollama pull llama3.2
        return OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # Ollama doesn't need a real key
        )
    
    elif provider == "groq":
        # Groq has a free tier - get key at https://console.groq.com
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        return OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
        )
    
    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return OpenAI(api_key=api_key)
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


# Default provider and model - change these to switch providers
DEFAULT_PROVIDER = "ollama"
DEFAULT_MODELS = {
    "ollama": "llama3.2",           # or "mistral", "qwen2.5", "gemma2"
    "groq": "llama-3.1-8b-instant", # free and fast
    "openai": "gpt-4o-mini",        # paid
}


def call_llm(
    prompt: str,
    system_prompt: str = "",
    model: str = None,
    provider: str = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """
    Call the LLM with a prompt and return the response text.
    
    Args:
        prompt: The user prompt to send
        system_prompt: Optional system-level instructions
        model: The model identifier (defaults based on provider)
        provider: One of "ollama", "groq", "openai" (default: ollama)
        temperature: Sampling temperature (0 = deterministic, 1 = creative)
        max_tokens: Maximum tokens in the response
    
    Returns:
        The model's response as a string
    """
    provider = provider or DEFAULT_PROVIDER
    model = model or DEFAULT_MODELS.get(provider, "llama3.2")
    
    client = get_client(provider)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    return response.choices[0].message.content.strip()
