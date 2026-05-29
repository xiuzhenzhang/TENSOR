from src.TPP.model.sahp_text.model import SAHPWrapper


def get_model() -> SAHPWrapper:
    """Teach the trainer and evaluator how to load the SAHP model.

    Returns:
        SAHPWrapper: The model.
    """
    return SAHPWrapper
