"""
Use LLMs to verify if one user is IO or not based on their timeline.
This file only generate and store LLM responses.

Please note that we only use local LLMs in our work.
LLM API calls might work but untested.
"""

import argparse
import gc
import importlib
import math
import os
import pickle as pkl
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Environment configuration
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["SGLANG_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
os.environ["SAFETENSORS_FAST_GPU"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import sglang as sgl
import torch
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from transformers import AutoTokenizer

# Constants
BATCH_SIZE = 10000
DEFAULT_FIGURE_KWARGS = {
    "font.size": 18,
    "figure.figsize": (8, 4),
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times",
    "mathtext.fontset": "dejavusans",
    "text.latex.preamble": "\\usepackage{amsmath}",
}

# Global variable for multiprocessing worker
_tokenizer_for_length = None

# Argument parser
parser = argparse.ArgumentParser(description="LLM Scoring for Dataset (SGLang version)")
parser.add_argument("--llm_name", type=str, default=None, help="The LLM name to use.")
parser.add_argument("--temperature", type=float, default=0.5, help="The temperature for LLM scoring.")
parser.add_argument("--prompt_version", type=str, help="The version of the selected prompt.")


def llm_score(version: int, *args, **kwargs) -> Any:
    """Route to the appropriate LLM scoring function based on version.

    Args:
        version: Version number (0 for embedding-based, 1 for prompt-based)
        *args: Positional arguments to pass to the scoring function
        **kwargs: Keyword arguments to pass to the scoring function

    Returns:
        Output from the selected scoring function
    """
    function_selector = {0: llm_score_ephad, 1: llm_score_prompt}
    return function_selector[version](*args, **kwargs)


def llm_score_prompt(
    llm: sgl.Engine,
    user_text: List[Dict[str, str]],
    barrier: List[int],
    max_gen_token_length: int,
    local_llm: bool,
    llm_name: str,
    param: Dict[str, Any],
    tokenizer: AutoTokenizer,
) -> List[str]:
    """Score users using prompt-based LLM generation.

    Args:
        llm: The SGLang engine
        user_text: List of user post messages in chat format
        barrier: Index boundaries for each user's post sequence
        max_gen_token_length: Maximum tokens to generate
        local_llm: Whether using a local LLM
        llm_name: Name of the LLM model
        param: Model parameters configuration
        tokenizer: The tokenizer for the model

    Returns:
        List of generated responses
    """
    additional_params = {} if param.get(llm_name) is None else param.get(llm_name).get("sample_param", {})

    sampling_params = {"max_new_tokens": max_gen_token_length}
    sampling_params.update(additional_params)

    # Format inputs using chat template
    formatted_input = []
    for item in user_text:
        formatted_input.append(
            tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True, return_dict=False)
        )

    # Generate outputs
    if 'n' in sampling_params:
        del sampling_params['n']
    print(sampling_params)
    outputs = llm.generate(formatted_input, sampling_params)
    outputs = [item["text"] for item in outputs]

    # Parse reasoning if needed
    outputs_without_thinking = []
    if param.get(llm_name, {}).get("reasoning_parser") is not None:
        parser_instance = ReasoningParser(param[llm_name].get("reasoning_parser"))
        for item in outputs:
            _, text = parser_instance.parse_non_stream(item)
            outputs_without_thinking.append(text)
    else:
        outputs_without_thinking = outputs

    # Handle multiple samples per input
    if sampling_params.get("n") is not None and sampling_params["n"] > 1:
        n = sampling_params["n"]
        outputs_without_thinking = [outputs_without_thinking[i::n] for i in range(n)]

    return outputs_without_thinking if local_llm else [item.choices[0].text for item in outputs]


def llm_score_ephad(
    llm: sgl.Engine, user_text: List[str], barrier: List[int], temperature: float
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Score users using embedding-based similarity.

    Args:
        llm: The SGLang engine
        user_text: List of all user posts
        barrier: Index boundaries for each user's post sequence
        temperature: Temperature parameter for softmax

    Returns:
        Tuple of (abnormal scores, additional metrics dict)
    """
    from llm_info_sglang import description_abnormal_category, description_normal_category

    # Get embeddings for all user posts
    embeddings_objs = llm.get_embedding(user_text)
    user_embeddings = []
    dataset_size = len(barrier) - 1

    for index in range(dataset_size):
        selected_embeddings_object = embeddings_objs[barrier[index] : barrier[index + 1]]
        selected_embeddings = torch.tensor([item["embedding"] for item in selected_embeddings_object])
        user_embeddings.append(selected_embeddings.mean(dim=0))

    # Get category embeddings
    abnormal = llm.get_embedding(description_abnormal_category)
    normal = llm.get_embedding(description_normal_category)

    emb_abnormal_conclusion = torch.mean(torch.tensor([item["embedding"] for item in abnormal]), dim=0)
    emb_normal_conclusion = torch.mean(torch.tensor([item["embedding"] for item in normal]), dim=0)

    # Calculate similarity scores
    user_embeddings_stacked = torch.stack(user_embeddings, dim=0)
    llm_abnormal_score = torch.nn.functional.cosine_similarity(
        emb_abnormal_conclusion.unsqueeze(dim=0), user_embeddings_stacked, dim=-1
    )
    llm_normal_score = torch.nn.functional.cosine_similarity(
        emb_normal_conclusion.unsqueeze(dim=0), user_embeddings_stacked, dim=-1
    )

    # Compute final scores
    merged_score = torch.stack([llm_abnormal_score, llm_normal_score], dim=-1)
    y_score_llm_only = torch.nn.functional.softmax(merged_score, dim=-1)[:, 0] / temperature

    return y_score_llm_only, {"llm_abnormal_score": llm_abnormal_score, "llm_normal_score": llm_normal_score}


def _extract_field_value(data: List[Any], field_name: str, default_value: Any) -> Any:
    """Extract and validate a field value from data.

    Args:
        data: List of data values
        field_name: Name of the field being extracted
        default_value: Default value if field is invalid

    Returns:
        Extracted value or default value
    """
    if not data or (isinstance(data[0], float) and math.isnan(data[0])):
        return default_value

    unique_values = set(data)
    if len(unique_values) == 1:
        return unique_values.pop()

    # If multiple values, use the first one
    return data[0] if data else default_value


def get_batch_text(
    llm_type: str,
    input_data: Dict[str, List[Any]],
    tokenizer: AutoTokenizer,
    max_prompt_token_length: int,
    prompt_version: str,
) -> List[Dict[str, str]]:
    """Convert input data into formatted text batches for LLM processing.

    Args:
        llm_type: Type of LLM processing ("emb" or "gen")
        input_data: Dictionary containing post and account information
        tokenizer: Tokenizer for the model
        max_prompt_token_length: Maximum prompt token length
        prompt_version: Version of the prompt template to use

    Returns:
        List of formatted message dictionaries
    """
    if llm_type == "emb":
        return []

    if llm_type == "gen":
        from llm_info_sglang import (
            account_pattern,
            each_event_pattern,
            prompt_baseline,
            reply_event_pattern,
            repost_event_pattern,
        )

        # Extract account information
        account_info_dict = {
            "account_profile_description": _extract_field_value(
                input_data["account_profile_description"], "account_profile_description", "No profile available."
            ),
            "follower_count": int(_extract_field_value(input_data["follower_count"], "follower_count", 0)),
            "following_count": int(_extract_field_value(input_data["following_count"], "following_count", 0)),
            "account_creation_date": _extract_field_value(
                input_data["account_creation_date"], "account_creation_date", "Unknown Date"
            ),
        }

        system_role = prompt_baseline[prompt_version]
        account_profile = account_pattern.format(**account_info_dict)

        # Create iterator over data rows
        dict_iterator = (dict(zip(input_data.keys(), values)) for values in zip(*input_data.values()))

        merged_texts = []
        for dict_data in dict_iterator:
            post_text = "\nPost:\n"
            post_text += each_event_pattern.format(**dict_data)

            # Add reply information if available
            if not isinstance(dict_data["reply_post"], float):
                post_text += reply_event_pattern.format(**dict_data)

            # Add repost information if available
            if not isinstance(dict_data["repost_account_profile_description"], float):
                post_text += repost_event_pattern.format(**dict_data)

            messages = [
                {"role": "system", "content": system_role},
                {"role": "user", "content": account_profile + post_text},
            ]
            merged_texts.append(messages)

        return merged_texts

    return []


def _init_tokenizer_worker(tokenizer_name: str) -> None:
    """Initialize worker process for tokenization.

    Args:
        tokenizer_name: Name of the tokenizer to load
    """
    global _tokenizer_for_length
    _tokenizer_for_length = AutoTokenizer.from_pretrained(tokenizer_name)


def _tokenize_message(item: Dict[str, str]) -> List[int]:
    """Worker function to tokenize a single message.

    Args:
        item: Message dictionary in chat format

    Returns:
        Tokenized message as list of token IDs
    """
    return _tokenizer_for_length.apply_chat_template(item, tokenize=True, add_generation_prompt=True, return_dict=False)


def calculate_ap_random_chunk(args: Tuple[int, int, np.ndarray, int]) -> List[float]:
    """Calculate average precision for random scores in parallel.

    Args:
        args: Tuple of (seed, n_iters, y_label, size)

    Returns:
        List of average precision scores
    """
    seed, n_iters, y_label, size = args
    rng = np.random.default_rng(seed)
    ap_random = []
    for _ in range(n_iters):
        random_scores = rng.uniform(low=0.0, high=1.0, size=size)
        ap_random.append(average_precision_score(y_true=y_label, y_score=random_scores, pos_label=0))
    return ap_random


def dump_to_pkl(data: Any, filepath: Path, compression: Optional[str] = None) -> int:
    """Save data to a pickle file with optional compression.

    Args:
        data: Data to save
        filepath: Path to save the file
        compression: Compression method ("lzma", "bz2", "gz", or None)

    Returns:
        0 on success
    """
    compression_algorithms = {
        None: open,
        "lzma": importlib.import_module("lzma").open,
        "bz2": importlib.import_module("bz2").open,
        "gz": importlib.import_module("gzip").open,
    }

    if compression is not None:
        filepath = Path(str(filepath) + f".{compression}")

    selected_open_function = compression_algorithms[compression]
    with selected_open_function(filepath, "wb") as f:
        pkl.dump(data, f)

    return 0


def load_from_pkl(filepath: Path, compression: Optional[str] = None) -> Any:
    """Load data from a pickle file with optional compression.

    Args:
        filepath: Path to the pickle file
        compression: Compression method ("lzma", "bz2", "gz", or None)

    Returns:
        Loaded data
    """
    compression_algorithms = {
        "lzma": importlib.import_module("lzma").open,
        "bz2": importlib.import_module("bz2").open,
        "gz": importlib.import_module("gzip").open,
    }

    if compression is None:
        # Auto-detect compression from extension
        selected_open_function = compression_algorithms.get(Path(filepath).suffixes[-1][1:], open)
    else:
        selected_open_function = compression_algorithms[compression]

    with selected_open_function(filepath, "rb") as f:
        data = pkl.load(f)

    return data


def save_fig(fig: plt.Figure, file_location: Path, file_name: str) -> None:
    """Save a matplotlib figure and clean up resources.

    Args:
        fig: Matplotlib figure to save
        file_location: Directory to save the figure
        file_name: Name of the file
    """
    fig.savefig(Path(file_location, file_name), bbox_inches="tight")
    fig.clear()
    plt.close(fig=fig)
    del fig
    gc.collect()


def write_to_txt(strings: Union[str, List[str]], filepath: Path) -> int:
    """Write strings to a text file.

    Args:
        strings: String or list of strings to write
        filepath: Path to the output file

    Returns:
        0 on success
    """
    with Path(filepath).open("w") as f:
        if isinstance(strings, list):
            f.writelines(f"{item}\n" for item in strings)
        else:
            f.write(strings)

    return 0


def main() -> None:
    """Main execution function."""
    args = parser.parse_args()

    # Configuration
    root_path = Path(__file__).parent.parent.parent.resolve()
    result_dir = "results"
    result_indexes = ["1", "2", "3"]
    procedure_name = "TPP"
    dataset_names = ["Egypt", "Russia_1", "China_1", "Iran_1", "UAE"]
    # dataset_names = ["Russia_1", "China_1", "Iran_1", "UAE"]
    task_name = "nll_with_label"
    seed = 42

    # Model configuration
    model_name = "sahp"
    lr = 0.002
    batch_size = 32
    nts = 20000
    train_procedure_config = None
    train_dataloader_config = "sahp_dl.yml"
    train_model_config = "sahp.yml"

    evaluate_procedure_config = None
    evaluate_dataloader_config = "sahp_ldl.yml"
    evaluate_task_config = None

    # LLM configuration
    do_llm = True
    max_gen_token_length = 10000
    api_key = "sk-or-v1-<YOUR_KEY>"

    if do_llm:
        from llm_info_sglang import param

        assert args.llm_name is not None, "Please provide the llm_name argument."

    llm_scoring_function = 1
    key_used_in_result = [
        "text",
        "post_time",
        "account_profile_description",
        "follower_count",
        "following_count",
        "account_creation_date",
        "reply_post",
        "reply_post_time",
        "reply_account_profile_description",
        "reply_follower_count",
        "reply_following_count",
        "reply_account_creation_date",
        "repost_post_time",
        "repost_account_profile_description",
        "repost_follower_count",
        "repost_following_count",
        "repost_account_creation_date",
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    local_llm = args.llm_name in param
    llm_type = "emb" if llm_scoring_function == 0 else "gen"

    dataset_files = ["evaluation_nll.pkl.bz2"]
    model_dir = f"results_{model_name}_lr{lr}_bs{batch_size}_nts{nts}_{train_procedure_config}_{train_dataloader_config}_{train_model_config}"
    task_conf_dir = f"{evaluate_procedure_config}_{evaluate_dataloader_config}_{evaluate_task_config}"

    # Determine maximum input length
    print("Deciding context length...")
    max_input_len = 5000

    # Initialize LLM
    if do_llm:
        if local_llm:
            additional_params = (
                {} if param.get(args.llm_name) is None else param.get(args.llm_name).get("create_param", {})
            )

            if llm_type == "emb":
                llm = sgl.Engine(
                    model_path=args.llm_name,
                    context_length=max_input_len + max_gen_token_length + 500,
                    random_seed=seed,
                )
            else:
                llm = sgl.Engine(
                    model_path=args.llm_name,
                    context_length=max_input_len + max_gen_token_length + 500,
                    random_seed=seed,
                    mem_fraction_static=0.8,
                    device="cuda",
                    **additional_params,
                )
        else:
            from vllm_online import CustomOpenAIforVLLM

            llm = CustomOpenAIforVLLM(
                base_url="https://openrouter.ai/api/", model=args.llm_name, device="cuda", api_key=api_key
            )

        llm_assigned_name = args.llm_name.replace("/", "_")

    # Process datasets
    for sampled_dataset in dataset_names:
        print(f"Processing {sampled_dataset}...")

        for sampled_dataset_file in dataset_files:
            for result_index in result_indexes:
                target_result_dir = (
                    root_path
                    / result_dir
                    / procedure_name
                    / result_index
                    / sampled_dataset
                    / model_dir
                    / task_name
                    / task_conf_dir
                )

                results = load_from_pkl(target_result_dir / sampled_dataset_file)

                user_text = []
                barrier = [0]
                y_score, y_label = [], []

                results_for_prompt = {
                    key: results[key] for key in ["nll_per_seq", "labels", "mask"] + key_used_in_result
                }
                dict_list = [
                    dict(zip(results_for_prompt.keys(), values)) for values in zip(*results_for_prompt.values())
                ]

                # Process data sequentially
                for dict_data in tqdm(
                    dict_list, desc=f"Assembling prompts using posts in {sampled_dataset}/{sampled_dataset_file}"
                ):
                    y_s = dict_data["nll_per_seq"]
                    y_l = dict_data["labels"]
                    mask_sum = sum(dict_data["mask"]) - 1
                    del dict_data["labels"], dict_data["nll_per_seq"], dict_data["mask"]
                    u_text = get_batch_text(llm_type, dict_data, tokenizer, 1, args.prompt_version)

                    y_score.append(y_s)
                    y_label.append(y_l)
                    barrier.append(barrier[-1] + mask_sum)
                    user_text.append(u_text)

                # Create output directory
                output_dir = root_path / f"useful_utilities/io_dataset/results/{sampled_dataset}/{result_index}"
                output_dir.mkdir(parents=True, exist_ok=True)

                if do_llm:
                    # Flatten user text and rebuild barriers
                    barrier = [0]
                    flatten_user_text = []
                    for item in user_text:
                        flatten_user_text.extend(item)
                        barrier.append(barrier[-1] + len(item))

                    # Get LLM responses
                    if llm_scoring_function == 1:
                        responses = llm_score_prompt(
                            llm,
                            flatten_user_text,
                            barrier,
                            max_gen_token_length,
                            local_llm,
                            args.llm_name,
                            param,
                            tokenizer,
                        )
                    else:
                        responses = llm_score_ephad(llm, flatten_user_text, barrier, args.temperature)

                    # Rearrange responses by user
                    final_responses = []
                    for idx in range(len(barrier) - 1):
                        final_responses.append(responses[barrier[idx] : barrier[idx + 1]])

                    # Save results
                    dump_to_pkl(
                        {
                            "responses": final_responses,
                            "y_label": y_label,
                            "y_score": y_score,
                        },
                        output_dir / f"llm_baseline_AP_{llm_assigned_name}_{sampled_dataset_file}.pkl",
                        compression="lzma",
                    )


if __name__ == "__main__":
    main()
