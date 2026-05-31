# TENSOR

This is the codebase of Temporal-bEhavior-laNguage Signals for information Operation Recognition (TENSOR), an unsupervised anomaly detection approach of information operation users, accepted by ECML/PKDD 2026.

## How to reproduce the results reported in our paper.

All commands below assume they are run from the repository root unless otherwise noted.

1. Install `uv` and create the Python environment.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.13
uv sync --no-default-groups --group base --group llm
```

2. Train and evaluate SAHP on the IO datasets.

The training jobs are defined in `parameter_set/TPP/sahp_parameter_set.py`. They train SAHP with learning rate `0.002`, batch size `32`, `20000` training steps, and three repeats, then evaluate `nll_with_label` on the IO datasets. The expected SAHP outputs are written under paths of this form:

```text
results/TPP/<repeat_index>/<dataset>/results_sahp_lr0.002_bs32_nts20000_None_sahp_dl.yml_sahp.yml/nll_with_label/None_sahp_ldl.yml_None/
```

The command for training and evaluating SAHP models

```bash
model_name=sahp
gpu="0"

uv run bulk_task_runner.py \
       --procedure_name TPP \
       --model $model_name \
       --job_name train_and_evaluate_on_io_datasets \
       --GPU $gpu \
       --interpreter "uv" "run"
```

and SAHP with text models. Please note that you need to use `useful_utilities/io_dataset/0.attach_embedding_to_data.py` to generate embeddings of tweet posts before training the SAHP with text models because this variant expects embedded dataset files named `emb_train`, `emb_evaluate`, and `emb_test`, as configured in `parameter_set/TPP/sahp_text_parameter_set.py`

```bash
model_name=sahp_text
gpu="0"

uv run bulk_task_runner.py \
       --procedure_name TPP \
       --model $model_name \
       --job_name train_and_evaluate_on_io_datasets \
       --GPU $gpu \
       --interpreter "uv" "run"
```

3. Generate the reported IO-detection results with the utilities in `useful_utilities/io_dataset`.

Run these commands from `useful_utilities/io_dataset`:

```bash
cd useful_utilities/io_dataset

uv run 1.llm_baseline_sglang.py --llm_name openai/gpt-oss-120b --prompt_version v2
uv run 1.llm_baseline.py
```

The SGLang baseline script reads SAHP NLL files from `results/TPP/<repeat_index>/<dataset>/.../evaluation_nll.pkl.bz2` and writes stored LLM responses under `useful_utilities/io_dataset/results/<dataset>/<repeat_index>/`. The model parameters, tensor parallel sizes, prompt text, and labels used by these scripts are defined in `llm_info_sglang.py`.

Before running the scoring script, check the constants near the top of `2.llm_scores_using_stored_data.py`, especially `dataset_names`, `result_indexes`, `llm_name`, `used_scoring_model`, and `dataset_files`. They must match the stored response files in `useful_utilities/io_dataset/results/<dataset>/<repeat_index>/`. For a fresh run from `1.llm_baseline_sglang.py`, also make sure the generated filenames and the `target_files` pattern in `2.llm_scores_using_stored_data.py` use the same LLM name and the same `evaluation_nll.pkl.bz2` or `test_nll.pkl.bz2` split.

Then run the TENSOR scoring utility on stored LLM responses:

```bash
uv run 2.llm_scores_using_stored_data.py
```

This script combines SAHP NLL scores with LLM response scores from `BAAI/bge-reranker-v2-m3`, writes per-repeat precision-recall plots and metric files under `useful_utilities/io_dataset/results/<dataset>/<repeat_index>/`, and writes aggregated AP/precision/recall/F1/AUC summaries under `useful_utilities/io_dataset/results/<dataset>/`.

For temperature-sensitivity experiments, edit `temperatures` in `2.llm_scores_using_stored_data.py` from `[0.5]` to `np.arange(0.1, 2.05, 0.05)`, rerun the scoring script, then run:

```bash
uv run best_temp.py
```

This writes `useful_utilities/io_dataset/temperature_rankings_gpt_oss.txt`.

Optional summary utilities are also available in the same folder:

```bash
uv run 5.average_precision.py
uv run 99.nll_distribution_per_label.py
```

These scripts compute average precision from SAHP NLL files and plot per-label NLL distributions. They contain hardcoded `root_path` values, so update those paths to this repository root before running them in a new checkout.


### Bibliography

TBD
