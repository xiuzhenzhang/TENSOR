def easy_model_load(procedure_name, *args, **kwargs):
    import importlib

    module = importlib.import_module(f"src.{procedure_name}")
    return module.easy_model_load(*args, **kwargs)


if __name__ == "__main__":
    root_path = "/home/undesired/coderepo/workflow_next"

    """
    easy_model_load for TPP.
    root_path = root_path
    replace_idx = ''
    dataset_name = 'hawkes_1_v2'
    dataset_name_in_model_config = 'syn',
    device = 'cuda:0'
    compile = False,
    evaluation = True,
    model_name = 'sahp'
    lr = '0.002'
    used_batch_size = 128,
    n_training_steps = 50000
    used_procedure_config = None
    used_dataloader_config = None
    model_config = 'sahp.yml'
    """
    model = easy_model_load(
        "TPP",
        False,
        root_path,
        "",
        "hawkes_1_v2",
        "syn",
        "cuda:0",
        compile=True,
        evaluation=True,
        only_model_structure=False,
        model_name="sahp",
        lr=0.002,
        used_batch_size=128,
        n_training_steps=50000,
        used_procedure_config=None,
        used_dataloader_config=None,
        model_config="sahp.yml",
    )

    print("stop here.")
