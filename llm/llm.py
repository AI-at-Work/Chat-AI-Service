import logging
import requests
import openai
import os
from typing import Optional

class LLM:
    def __init__(self, openai_api_key):
        if not openai_api_key:
            raise ValueError("API key is required for OpenAI")
        self.openai = openai.OpenAI(
            # This is the default and can be omitted
            api_key=openai_api_key,
        )

    def generate(self, provider, model, user_prompt, system_prompt="", max_tokens=None):
        logging.info("Generating LLM... for provider " + provider)
        if provider == "ollama":
            return self._generate_ollama(model, user_prompt, system_prompt, max_tokens)
        elif provider == "openai":
            return self._generate_openai(model, user_prompt, system_prompt, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _generate_ollama(self, model, user_prompt, system_prompt, max_tokens):
        ollama_port = os.getenv("OLLAMA_PORT")
        url = str(f"http://ollama:{ollama_port}/api/generate")
        logging.info("ollama url: " + url)
        data = {
            "model": model,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False
        }
        if max_tokens is not None:
            data["max_tokens"] = max_tokens

        try:
            response = requests.post(url, json=data, timeout=10)  # Add a timeout
            response.raise_for_status()  # Raise an exception for bad status codes

            response_json = response.json()
            return {
                "text": response_json["response"],
                "input_tokens": response_json["prompt_eval_count"],
                "output_tokens": response_json["eval_count"]
            }
        except ConnectionError as e:
            raise Exception(f"Failed to connect to Ollama API. Is the server running? Error: {str(e)}")
        except requests.exceptions.Timeout:
            raise Exception("Request to Ollama API timed out. The server might be overloaded or not responding.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error communicating with Ollama API: {str(e)}")

    def _generate_openai(self, model, user_prompt, system_prompt, max_tokens: Optional[int]):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = self.openai.chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens
        )

        return {
            "text": response.choices[0].text,
            "input_tokens": int(response.usage.prompt_tokens),
            "output_tokens": int(response.usage.completion_tokens)
        }
