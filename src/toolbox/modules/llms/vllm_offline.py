import re

from vllm import LLM, SamplingParams


class VLLMOfflineInference:
    def __init__(self, model, device, model_args):
        """
        Access an vLLM service endpoint.

        ### Args
            * ```str``` model
              The name of the model to use.
            * ```str``` device
              The device to run the model on (e.g., 'cpu', 'cuda').
            * ```dict``` model_args
              Additional LLM parameter when loading the model.
        """
        self.device = device
        self.model = model
        quantization_keywords = ["awq", "gptq"]

        if not any(s in self.model.lower() for s in quantization_keywords) and model_args["quantization"] is not None:
            print("Quantization set but the loaded LLM is not quantized!")
            print("Something is wrong? Anyway, we will ignore the quantization config so we can continue.")
            del model_args["quantization"]

        self.llm = LLM(self.model, **model_args)
        self.tokenizer = self.llm.get_tokenizer()

    def completion(self, prompt, stream=False, token=True, **kwargs):
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
        sampling_params = SamplingParams(**kwargs)

        if token:
            result = self.llm.generate({"prompt_token_ids": prompt}, sampling_params=sampling_params)
        else:
            result = self.llm.generate(prompt, sampling_params)

        return result

    def completions(self, prompt_list, stream=False, n_threads=1, token=True, **kwargs):
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

        sampling_params = SamplingParams(**kwargs)

        if token:
            token_strings = prompt_list
        else:
            token_strings = self.tokenize(prompt_list)

        packed_tokens = []
        for token_string in token_strings:
            packed_tokens.append({"prompt_token_ids": token_string})

        results = self.llm.generate(packed_tokens, sampling_params, use_tqdm=False)

        return results

    def tokenize(self, inputs):
        if isinstance(inputs, list):
            token_ids = []
            for input_ in inputs:
                token_ids.append(self.tokenizer.encode(input_))
        else:
            token_ids = self.tokenizer.encode(inputs)
        return token_ids

    def detokenize(self, inputs):
        if isinstance(inputs, list):
            token_ids = []
            for input_ in inputs:
                token_ids.append(self.tokenizer.decode(input_))
        else:
            token_ids = self.tokenizer.decode(inputs)
        return token_ids


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


if __name__ == "__main__":
    model_name = "Qwen/Qwen3-0.6B"
    model = VLLMOfflineInference(
        model_name,
        device="cuda:0",
        model_args={
            "quantization": "awq_marlin",
            "max_model_len": 2707,
            "gpu_memory_utilization": 0.6,
            "enable_prefix_caching": True,
        },
    )

    # model_kwargs = {'temperature': 0.8},
    # pipeline_kwargs = {'max_new_tokens': 50},
    # token_kwargs = {'model_max_length': 16384, 'truncation': True})

    # echo.
    prompt = [
        "You are a helpful assistant.",
        "You are a helpful assistant.",
        "You are a helpful assistant.",
        "You are a helpful assistant.",
        "You are a helpful assistant.",
        "You are a helpful assistant.",
        "You are a helpful assistant.",
        "You are a helpful assistant.",
    ]
    tokens = model.tokenize(prompt)

    response = model.completions(tokens, prompt_logprobs=1, max_tokens=0, temperature=0.0)

    probs_per_batch = []
    for idx, output in enumerate(response):
        raw_recorded_logprobs = output.prompt_logprobs[-3:]
        probs_per_seq = []
        for item in raw_recorded_logprobs:
            probs_per_seq.append(list(item.values())[0].logprob)
        probs_per_batch.append(probs_per_seq)

    print(f"Direct response: {response}")
    print(f"Extracted probability: {probs_per_batch}")
    # print('----------------------------------------------------------')
    # print(extract_content(response))
    # print('----------------------------------------------------------')
    # print(remove_thinking(extract_content(response)))
    #
    # messages = [message,] * 4
    # responses = model.chats(messages, logprobs = True, top_logprobs = 3, temperature = 0.6)

    # print(responses)
    # print('----------------------------------------------------------')
    # print(extract_content(responses))
    # print('----------------------------------------------------------')
    # outputs = remove_thinking(extract_content(responses))
    # for output in outputs:
    #     print(output)
    #     print('------------------------End------------------------')
