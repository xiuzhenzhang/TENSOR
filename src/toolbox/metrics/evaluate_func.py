# All evaluate functions used in evaluate_on_batch().


def microf1(input_data, target, **kwargs):
    from sklearn.metrics import f1_score

    return f1_score(y_pred=input_data, y_true=target, average="micro", zero_division=1.0)


def macrof1(input_data, target, **kwargs):
    from sklearn.metrics import f1_score

    return f1_score(y_pred=input_data, y_true=target, average="macro", zero_division=1.0)


def acc(input_data, target, **kwargs):
    from sklearn.metrics import accuracy_score

    return accuracy_score(y_pred=input_data, y_true=target)


def top_k(input_data, target, **kwargs):
    import numpy as np
    from sklearn.metrics import accuracy_score, top_k_accuracy_score

    num_events = input_data.shape[-1]
    top_k_acc_single_event_seq = []
    if num_events > 2:
        for k in range(1, num_events):
            top_k_acc_single_event_seq.append(
                top_k_accuracy_score(y_true=target, y_score=input_data, k=k, labels=np.arange(num_events))
            )
    else:
        top_k_acc_single_event_seq.append(accuracy_score(y_true=input_data, y_pred=target))

    return top_k_acc_single_event_seq


def l1(input_data, target, timestamp, **kwargs):
    from src.toolbox.metrics import L1_distance_between_two_funcs

    return L1_distance_between_two_funcs(input_data, target, timestamp)


def spearman(input_data, target, **kwargs):
    from scipy.stats import spearmanr

    return spearmanr(input_data.flatten(), target.flatten())[0]


evaluate_func_dict = {
    "acc": acc,
    "macro-f1": macrof1,
    "micro-f1": microf1,
    "top_k": top_k,
    "l1": l1,
    "spearman": spearman,
}


def evaluate_func(name):
    return evaluate_func_dict[name]
