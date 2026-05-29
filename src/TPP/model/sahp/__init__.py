from src.TPP.model.sahp.model import SAHPWrapper


def get_model() -> SAHPWrapper:
    """Teach the trainer and evaluator how to load the SAHP model.

    Returns:
        SAHPWrapper: The model.
    """
    return SAHPWrapper
