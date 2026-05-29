import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin

import requests
from openai import OpenAI


class CustomOpenAIforVLLM:
    def __init__(self, base_url, model, device, api_key):
        """
        Access an vLLM service endpoint.

        ### Args
            * ```str``` base_url
              The path to the API endpoint.
            * ```str``` model
              The name of the model to use.
            * ```str``` device
              The device to run the model on (e.g., 'cpu', 'cuda').
            * ```str``` api_key = 'ollama'
              The API key for accessing the vLLM services.
        """
        self.base_url = base_url  # Store for multiprocessing
        self.device = device
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(base_url=urljoin(base_url, "v1"), api_key=api_key)

    def completion(self, prompt, stream=False, **kwargs):
        """
        Send a chat request to the vLLM language model.

        ### Args
            * ```list[dict]``` message
              The conversation history and current query in OpenAI's chat format.

            * ```bool``` stream
              Whether to stream the response incrementally.

        ### Outputs
            * ```Completion``` completions
              Structure:
                id: response-id
                choices: a list containing Choice. Usually its length is 1.
                  Choice:
                    finish_reason: why the output process finished, because the generation stops or the output exceeds the output limit.
                    index: output index.
                    logprobs: log of the probability of the generated sequence.
                    message: the content of the response. It is a ChatCompletionMessage.
                    ChatCompletionMessage:
                      content: the response
                      refusal: <unknown>
                      role: The role of this response. System: global settings. User: the user input. Assistant: the model output.
                      annotations: <unknown>
                      audio: <unknown>
                      function_call: <unknown>
                      tool_calls: <unknown>
                    created: the time of when this response is created recorded in UNIX timestamp
                    model: the language model used to generate the response
                    object: The name of the used API endpoint
                    service_tier: <unknown>
                    system_fingerprint: 'fp_ollama' means ollama
                    usage: statistics of this response. This item is a CompletionUsage object.
                    CompletionUsage:
                      completion_tokens: length of the generated sequence counted in the number of tokens.
                      prompt_tokens: length of the prompt sequence counted in the number of tokens.
                      total_tokens: length of the prompt and generated sequence counted in the number of tokens.
                      completion_tokens_details: <unknown>
                      prompt_tokens_details: <unknown>

              The API response containing the generated text or error details.
        """
        return self.client.completions.create(model=self.model, prompt=prompt, stream=stream, **kwargs)

    def generate(self, prompt_list, params, stream=False, n_threads=4, **kwargs):
        """
        Send multiple chat requests in parallel using multiprocessing.

        ### Args
            * ```list[list[dict]]``` messages_list
              A list of conversation histories, each in OpenAI's chat format.
            * ```bool``` stream
              Whether to stream the response incrementally.

        ### Outputs
            * ```list[Completion]```
              A list of responses for each input message in the same order.
        """
        # Disable streaming for multiprocessing as it's not supported
        if stream:
            raise NotImplementedError("Streaming not supported for multiprocessing")

        # Use multiprocessing Pool to process requests in parallel
        if n_threads > 1:
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                results_in_future = {
                    idx: executor.submit(self.completion, prompt, stream, **kwargs)
                    for idx, prompt in enumerate(prompt_list)
                }

            results = []
            prompt_list_length = len(prompt_list)
            for idx in range(prompt_list_length):
                try:
                    results.append(results_in_future[idx].result())
                except Exception:
                    results.append(None)
        else:
            results = []
            for message in prompt_list:
                results.append(self.completion(message, stream, **kwargs))

        return results

    def tokenize(self, input_seq, full_output=False):
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        payload = {"model": self.model, "prompt": input_seq}

        response = requests.post(f"{self.base_url}/tokenize", headers=headers, json=payload)
        response = response.json()

        return response if full_output else response["tokens"]

    def detokenize(self, input_seq, full_output=False):
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        payload = {"model": self.model, "prompt": input_seq}

        response = requests.post(f"{self.base_url}/detokenize", headers=headers, json=payload)
        response = response.json()

        return response if full_output else response["tokens"]


def extract_content(completions):
    if isinstance(completions, list):
        contents = []

        for completion in completions:
            for choice in completion.choices:
                contents.append(choice.message.content)
    else:
        contents = []

        for choice in completions.choices:
            contents.append(choice.message.content)

    return contents


def remove_thinking(content):
    if isinstance(content, list):
        cleaned_content = [re.sub(r"<think>.*?</think>\n?", "", item, flags=re.DOTALL) for item in content]
    elif isinstance(content, str):
        cleaned_content = re.sub(r"<think>.*?</think>\n?", "", content, flags=re.DOTALL)
    else:
        raise Exception(f"Unknown input {content}.")

    return cleaned_content


def create_messages(system=None, **kwargs):
    messages = [{"role": "system", "content": system}] if system is not None else []
    for key, value in kwargs.items():
        striped_key = key.strip("0123456789")
        if striped_key not in ["user", "assistant"]:
            raise Exception(f"Unknown role {key} detected! Available roles are user and assistant.")
        messages.append({"role": striped_key, "content": value})

    return messages
