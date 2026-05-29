import gc
import multiprocessing as mp
from pathlib import Path

import sglang as sgl
import yaml


'''
This file embeds retweet posts of IO datasets into sentence embeddings.
'''


def load_from_pkl(filepath, compression=None):
    import importlib
    import pathlib
    import pickle as pkl

    dict_compression_algorithms = {
        # Is it a good choice?
        "lzma": importlib.import_module("lzma").open,
        "bz2": importlib.import_module("bz2").open,
        "gz": importlib.import_module("gzip").open,
    }

    # If compression is None, we will guess the compression algorithm.
    # ***.lzma -> lzma
    # ***.bz2 -> bz2
    # ***.gz -> gz
    # others -> no compression.
    if compression is None:
        selected_open_function = dict_compression_algorithms.get(pathlib.Path(filepath).suffixes[-1][1:], open)
    else:
        selected_open_function = dict_compression_algorithms[compression]

    f = selected_open_function(filepath, "rb")
    data = pkl.load(f)
    f.close()

    return data


def dump_to_pkl(data, filepath, compression=None):
    import importlib
    import os
    import pickle as pkl
    from pathlib import Path

    dict_compression_algorithms = {
        None: open,
        # Is it a good choice?
        "lzma": importlib.import_module("lzma").open,
        "bz2": importlib.import_module("bz2").open,
        "gz": importlib.import_module("gzip").open,
    }
    """
    Add proper suffix to the base file name if compression is not None.
    """
    head, tail = os.path.split(filepath)
    tail = tail + f"{'.' + compression if compression is not None else ''}"
    filepath = Path(head, tail)

    selected_open_function = dict_compression_algorithms[compression]
    f = selected_open_function(filepath, "wb")
    pkl.dump(data, f)
    f.close()

    return 0


def process_dataset_with_engine(sampled_dataset, root, embedding_model, dim, engine_id, dp_size, batch_size=50000):
    """
    Process a single dataset using one dedicated engine (one GPU).
    Each engine has dp_size=1 for single GPU operation.
    """
    import os

    # Set CUDA_VISIBLE_DEVICES to assign specific GPU to this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(engine_id % dp_size)

    print(f"[Engine {engine_id}] Starting engine with dp_size=1 for {sampled_dataset} on GPU {engine_id % dp_size}...")

    # Initialize SGLang engine with dp_size=1 for this instance (one GPU)
    llm = sgl.Engine(
        model_path=embedding_model,
        is_embedding=True,
        context_length=3050,
        mem_fraction_static=0.9,
        trust_remote_code=True,
        dp_size=1,  # Each engine uses only 1 GPU
        port=30000 + engine_id,
        max_running_requests=50,  # Very low to prevent OOM on large datasets
    )

    print(f"[Engine {engine_id}] Processing {sampled_dataset}...")

    dataset_dir = root / sampled_dataset

    # load the property card
    dataset_card_path = dataset_dir / "dataset_card.yml"
    with dataset_card_path.open("r") as f_property:
        dataset_card = yaml.safe_load(f_property)

    file_names = [dataset_dir / "train.pkl.lzma", dataset_dir / "evaluate.pkl.lzma", dataset_dir / "test.pkl.lzma"]

    for sampled_dataset_file in file_names:
        data = load_from_pkl(sampled_dataset_file)
        text_data = data["text"]

        barrier = [0]
        for each_line_of_data in text_data:
            barrier.append(barrier[-1] + len(each_line_of_data))

        # Collect all texts into a flat list (memory is not an issue)
        # Convert all items to strings to handle non-string inputs
        all_texts = []
        for each_line_of_data in text_data:
            for text in each_line_of_data:
                # Convert to string if not already, use placeholder for None/NaN
                if isinstance(text, str):
                    all_texts.append(text if text else " ")  # Replace empty strings with space
                elif text is None or (isinstance(text, float) and text != text):  # None or NaN
                    all_texts.append(" ")  # Use space as placeholder
                else:
                    all_texts.append(str(text))

        # Encode texts in batches to avoid out of memory
        print(f"[Engine {engine_id}] Encoding {len(all_texts)} texts for {sampled_dataset_file.name} in batches of {batch_size}...")
        text_embeddings = []

        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i+batch_size]
            batch_results = llm.encode(batch)
            text_embeddings.extend([item["embedding"][:dim] for item in batch_results])
            del batch_results
            gc.collect()
            if (i // batch_size + 1) % 10 == 0:
                print(f"[Engine {engine_id}] Processed {i+len(batch)}/{len(all_texts)} texts")

        del all_texts
        gc.collect()

        # Reshape embeddings back to original structure
        reshaped_text_embeddings = []
        for idx in range(len(text_data)):
            reshaped_text_embeddings.append(text_embeddings[barrier[idx] : barrier[idx + 1]])

        del text_embeddings
        del barrier

        data["text_emb"] = reshaped_text_embeddings
        target_dataset_file = sampled_dataset_file.parent / ("emb_" + sampled_dataset_file.name.rsplit(".", 1)[0])
        dump_to_pkl(data, target_dataset_file, compression="lzma")

        del data
        del text_data
        del reshaped_text_embeddings
        gc.collect()

    dataset_card["emb_dim"] = dim

    with dataset_card_path.open("w") as outfile:
        yaml.dump(dataset_card, outfile, default_flow_style=False, allow_unicode=True)

    print(f"[Engine {engine_id}] Completed {sampled_dataset}")
    llm.shutdown()
    return sampled_dataset


if __name__ == "__main__":
    # Number of parallel engine instances to run
    # Each engine uses dp_size=1 (one GPU per engine)
    # Instead of using dp_size=8, we start DP_SIZE engines each with dp=1
    DP_SIZE = 8

    root = Path("/home/ubuntu/workflow_next/data")
    datasets_dir = "TPP"
    datasets = [
        "Qatar",
        "Bangladesh",
        "Armenia",
        "Ecuador",
        "Egypt",
        "Russia_1",
        "Thailand",
        "Catalonia",
        "China_1",
        "Iran_6",
        "Iran_1",
        "Ghana",
        "Russia_4",
        "Iran_2",
        "Iran_5",
        "Venezuela_1",
        "Spain",
        "China_2",
        "Iran_4",
        "Iran_3",
        "Russia_5",
        "Russia_2",
        "Cuba",
        "UAE",
    ]
    # datasets = ["Egypt", "Russia_1", "China_1", "Iran_1", "UAE"]
    embedding_model = "Qwen/Qwen3-Embedding-8B"
    # We only keep the first 128 dim for space saving.
    dim = 16
    # Batch size for encoding - adjust based on available memory
    # Very small batch size for large datasets (200k+ posts)
    # Must be small enough that max_running_requests can handle it
    batch_size = 50

    root = root / datasets_dir

    print(f"Starting {DP_SIZE} engine instances (each with dp_size=1)...")
    print(f"Each engine processes one dataset on one GPU")

    # Use a queue-based approach to keep GPUs busy
    # Start new tasks as soon as a GPU becomes available
    import time

    active_processes = {}  # {gpu_id: (process, dataset_name)}
    dataset_queue = list(enumerate(datasets))
    completed_datasets = []

    while dataset_queue or active_processes:
        # Start new processes on available GPUs
        for gpu_id in range(1, DP_SIZE):
            if gpu_id not in active_processes and dataset_queue:
                idx, dataset = dataset_queue.pop(0)
                p = mp.Process(
                    target=process_dataset_with_engine, args=(dataset, root, embedding_model, dim, gpu_id, DP_SIZE, batch_size)
                )
                p.start()
                active_processes[gpu_id] = (p, dataset)
                print(f"Started {dataset} on GPU {gpu_id}")

        # Check for completed processes and free up GPUs
        completed_gpus = []
        for gpu_id, (process, dataset_name) in active_processes.items():
            if not process.is_alive():
                process.join()
                completed_gpus.append(gpu_id)
                completed_datasets.append(dataset_name)
                print(f"Completed {dataset_name} on GPU {gpu_id}")

        # Remove completed processes
        for gpu_id in completed_gpus:
            del active_processes[gpu_id]

        # Small sleep to avoid busy-waiting
        if active_processes:
            time.sleep(0.1)

    print(f"All {len(completed_datasets)} datasets processed successfully: {completed_datasets}")
