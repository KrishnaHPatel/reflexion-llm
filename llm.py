"""
LLM API Wrapper

This module provides a simple interface to call language models.
This implementation uses Ollama only, running locally on your machine.
"""

import json
import urllib.error
import urllib.request


DEFAULT_MODEL = "llama3.2"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"


def call_llm(
    prompt: str,
    system_prompt: str = "",
    model: str = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """
    Call the LLM with a prompt and return the response text.
    
    Args:
        prompt: The user prompt to send
        system_prompt: Optional system-level instructions
        model: Ollama model name, such as "llama3.2" or "mistral"
        temperature: Sampling temperature (0 = deterministic, 1 = creative)
        max_tokens: Maximum tokens in the response
    
    Returns:
        The model's response as a string
    """
    model = model or DEFAULT_MODEL
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    request = urllib.request.Request(
        OLLAMA_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not connect to Ollama. Make sure Ollama is running and "
            f"the model is installed with: ollama pull {model}"
        ) from exc

    return response_data["message"]["content"].strip()
