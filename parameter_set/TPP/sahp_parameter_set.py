# parameter sets of model SAHP.

replace = False
wandb = False
retrain = 1 if replace else 3
seed_during_evaluation = 12345

model_name = 'sahp'
dataloader_name = "generic"
model_dtype='float32'
compile_or_not=True

# Hyperparameters used to train on synthetic datasets.
syn_datasets_name = ["hawkes_1_v2", "hawkes_2_v2", "poisson_v2", "self_correct_v2", "stationary_renewal_v2"]
# syn_datasets_name = ["hawkes_1_v2",]
syn_n_training_step = 50000
syn_training_batch_size = 128
syn_evaluation_batch_size = 512
syn_learning_rate = 0.002
syn_training_dataloader_config = f'syn/{model_name}_dl.yml'
syn_model_config = f"syn/{model_name}.yml"
syn_n_warmup_steps = int(syn_n_training_step * 0.2)
syn_n_evaluation_steps = int(0.02 * syn_n_training_step)

# Additional hyperparameter used for evaluation.
syn_dataloader_config = "syn/plot.yml"

# Train MTPP models and evaluate them on synthetic datasets, such as hawkes_1, hawkes_2, poisson, self_correct, and stationary_renewal.
train_on_syn_datasets = {
    "worker": "start.py",
    "job_type": "train",
    "repeat": retrain,
    "static":
    {
        "no_seed": True,
        "dataloader_name": dataloader_name,
        "dataloader_config": syn_training_dataloader_config,
        "n_training_steps": syn_n_training_step,
        "n_evaluation_steps": syn_n_evaluation_steps,
        "n_report_steps": syn_n_evaluation_steps,
        "training_batch_size": syn_training_batch_size,
        "evaluation_batch_size": syn_evaluation_batch_size,
        "n_warmup_steps": syn_n_warmup_steps,
        "model_name": model_name,
        "lr": syn_learning_rate,
        "save_mode": "best",
        "lr_sched": True,
        "op_name": "AdamW",
        "optim_config": "optimizer.yml",
        "model_config": syn_model_config,
        "n_cycles": "0.5",
        "replace": replace,
        "wandb": wandb,
        "dtype": model_dtype,
        "compile": compile_or_not,
    },
    "sequential":
    {
        "dataset_name": syn_datasets_name,
    }
}

evaluate_on_syn_datasets = {
    "worker": "start.py",
    "job_type": "evaluate",
    "static": {
        "seed": seed_during_evaluation,
        "model_name": model_name,
        "lr": syn_learning_rate,
        "dataloader_name": dataloader_name,
        "n_training_steps": syn_n_training_step,
        "test_data_name": "test",
        "used_batch_size": syn_training_batch_size,
        "dataloader_config": syn_dataloader_config,
        "used_dataloader_config": syn_training_dataloader_config,
        "model_config": syn_model_config,
        "replace": replace,
        "dtype": model_dtype,
        "compile": compile_or_not,
        "combine_used_and_current_dataloader_config": True,
        },
    'sequential': {
        "dataset_name": syn_datasets_name,
    },
    "combinatorial": {
        'sequential':{
            "evaluation_batch_size": [1, 1, 1, 128, 128, 128],
            "task_name": ["intensity", "probability", "debug", "spearman_and_l1", "mae_and_f1", "mae_e_and_f1"],
            "task_config": ["syn/evaluate_intensity.yml", "syn/evaluate_probability.yml", \
                            "syn/evaluate_debug.yml", "syn/evaluate_l1.yml", \
                            "syn/evaluate_mae.yml", "syn/evaluate_mae_e.yml"]
        }
    }
}

# build tasks
train_and_evaluate_on_syn_datasets = [train_on_syn_datasets] + [evaluate_on_syn_datasets]

# Hyperparameters used to train and evaluate on real-world datasets.
realworld_dataset_name = ['bookorder', 'retweet', 'stackoverflow', 'taobao', 'usearthquake', 'yelp']
realworld_learning_rate = 0.002
realworld_training_dataloader_config = f'{model_name}_dl.yml'
realworld_training_dataloader_config = [f'{dataset}/{realworld_training_dataloader_config}' for dataset in realworld_dataset_name]
realworld_model_config = [f'{dataset}/{model_name}.yml' for dataset in realworld_dataset_name]
realworld_training_step = [20000, 400000, 200000, 80000, 80000, 200000]
realworld_n_evaluation_steps = [int(0.02 * training_step) for training_step in realworld_training_step]
realworld_n_report_steps = [int(0.02 * training_step) for training_step in realworld_training_step]
realworld_training_batch_size = [8, 32, 32, 32, 32, 32]
realworld_evaluation_batch_size = [8, 32, 32, 32, 32, 32]
realworld_n_warmup_steps = [int(0.2 * training_step) for training_step in realworld_training_step]

train_on_realworld_datasets = {
    "worker": "start.py",
    "job_type": "train",
    "repeat": retrain,
    "static":
     {
         "no_seed": True,
         "dataloader_name": dataloader_name,
         "model_name": model_name,
         "lr": realworld_learning_rate,
         "save_mode": "best",
         "lr_sched": True,
         "op_name": "AdamW",
         "optim_config": "optimizer.yml",
         "n_cycles": 0.5,
         "replace": replace,
         "wandb": wandb,
     },
     'sequential':
     {
         "dataset_name": realworld_dataset_name,
         "dataloader_config": realworld_training_dataloader_config,
         "model_config": realworld_model_config,
         "n_training_steps": realworld_training_step,
         "n_evaluation_steps": realworld_n_evaluation_steps,
         "n_report_steps": realworld_n_report_steps,
         "training_batch_size": realworld_training_batch_size,
         "evaluation_batch_size": realworld_evaluation_batch_size,
         "n_warmup_steps": realworld_n_warmup_steps,
     }
}

realworld_evaluation_dataloader_config = 'plot.yml'
realworld_evaluation_dataloader_config = [f'{dataset}/{realworld_evaluation_dataloader_config}' for dataset in realworld_dataset_name]
realworld_evaluation_task = ["intensity", "probability", "debug", "mae_and_f1", "mae_e_and_f1"]
realworld_config_files = ['evaluate_intensity.yml', 'evaluate_probability.yml', 'evaluate_debug.yml', 'evaluate_mae.yml', 'evaluate_mae_e.yml']
realworld_evaluation_task_config = [[f'{dataset}/{config}' for dataset in realworld_dataset_name] for config in realworld_config_files]

evaluate_on_realworld_datasets = {
    "worker": "start.py",
    "job_type": "evaluate",
    "static":
    {
        "seed": seed_during_evaluation,
        "model_name": model_name,
        "lr": realworld_learning_rate,
        "dataloader_name": dataloader_name,
        "test_data_name": "test",
        "replace": replace,
        "combine_used_and_current_dataloader_config": True,
    },
    'combinatorial':
    {
        'task_name': realworld_evaluation_task,
        'sequential':
        {
            "evaluation_batch_size": [1, 1, 1, 32, 32],
            "dataset_name": realworld_dataset_name,
            "dataloader_config": realworld_evaluation_dataloader_config,
            "used_dataloader_config": realworld_training_dataloader_config,
            "model_config": realworld_model_config,
            "n_training_steps": realworld_training_step,
            "used_batch_size": realworld_training_batch_size,
        },
        'value_matrices': {
            "task_config": realworld_evaluation_task_config
        }
    }
}

# build tasks
train_and_evaluate_on_realworld_datasets = [train_on_realworld_datasets] + [evaluate_on_realworld_datasets]

# Hyperparameters used to train and evaluate on IO datasets.
io_dataset_name = ['Cuba', 'Bangladesh', 'Qatar', 'Armenia', 'Ecuador', 'Egypt', 'Russia_1', 'Thailand', 'Catalonia', 'China_1', 'Iran_6', 'Iran_1', 'Ghana', 'Russia_4', 'Iran_2', 'Iran_5', 'Venezuela_1', 'Spain', 'China_2', 'Iran_4', 'Iran_3', 'Russia_5', 'Russia_2', 'UAE']
# io_dataset_name = ['Russia', 'Cuba']
io_learning_rate = 0.002
io_training_dataloader_config = f'{model_name}_dl.yml'
io_training_dataloader_config = f'io_dataset/{io_training_dataloader_config}'
io_model_config = f'io_dataset/{model_name}.yml'
io_training_step = 20000
io_n_evaluation_steps = int(0.02 * io_training_step)
io_n_report_steps = int(0.02 * io_training_step)
io_training_batch_size = 32
io_evaluation_batch_size = 32
io_n_warmup_steps = int(0.2 * io_training_step)

train_on_io_datasets = {
    "worker": "start.py",
    "job_type": "train",
    "repeat": retrain,
    "static":
     {
         "no_seed": True,
         "dataloader_name": dataloader_name,
         "model_name": model_name,
         "lr": io_learning_rate,
         "save_mode": "best",
         "lr_sched": True,
         "op_name": "AdamW",
         "optim_config": "optimizer.yml",
         "n_cycles": 0.5,
         "replace": replace,
         "wandb": wandb,
         "dtype": model_dtype,
         "compile": compile_or_not,
         "dataloader_config": io_training_dataloader_config,
         "training_batch_size": io_training_batch_size,
         "evaluation_batch_size": io_evaluation_batch_size,
         "model_config": io_model_config,
         "n_training_steps": io_training_step,
         "n_report_steps": io_n_report_steps,
         "n_warmup_steps": io_n_warmup_steps,
         "n_evaluation_steps": io_n_evaluation_steps,
     },
     'sequential':
     {
         "dataset_name": io_dataset_name,
     }
}

io_evaluate_dataloader_name = 'generic_label'
io_evaluation_dataloader_config = f'io_dataset/{model_name}_ldl.yml'
io_evaluation_task = ["nll_with_label"]

evaluate_on_io_datasets = {
    "worker": "start.py",
    "job_type": "evaluate",
    "static":
    {
        "seed": seed_during_evaluation,
        "model_name": model_name,
        "lr": io_learning_rate,
        "dataloader_name": io_evaluate_dataloader_name,
        "training_data_name": "train",
        "evaluate_data_name": "evaluate",
        "test_data_name": "test",
        "replace": replace,
        "combine_used_and_current_dataloader_config": True,
        "training_batch_size": 1,
        "evaluation_batch_size": 1,
        "used_dataloader_config": io_training_dataloader_config,
        "model_config": io_model_config,
        "n_training_steps": io_training_step,
        "used_batch_size": io_training_batch_size,
        "dataloader_config": io_evaluation_dataloader_config,
    },
    'combinatorial':
    {
        'task_name': io_evaluation_task,
        'sequential':
        {
            "dataset_name": io_dataset_name,
        },
    }
}

# build tasks
train_and_evaluate_on_io_datasets = [train_on_io_datasets] + [evaluate_on_io_datasets]

# Define the sahp hyperparameter list.

sahp_hyperparameter_list = {
    'train_on_syn_datasets': train_on_syn_datasets,
    'evaluate_on_syn_datasets': evaluate_on_syn_datasets,
    'train_and_evaluate_on_syn_datasets': train_and_evaluate_on_syn_datasets,

    'train_on_realworld_datasets': train_on_realworld_datasets,
    'evaluate_on_realworld_datasets': evaluate_on_realworld_datasets,
    'train_and_evaluate_on_realworld_datasets': train_and_evaluate_on_realworld_datasets,

    'train_on_io_datasets': train_on_io_datasets,
    'evaluate_on_io_datasets': evaluate_on_io_datasets,
    'train_and_evaluate_on_io_datasets': train_and_evaluate_on_io_datasets,
}
