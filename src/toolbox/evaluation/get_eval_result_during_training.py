# Several extensive operations for python list.
from tqdm import tqdm

from src.toolbox.list_operation import list_add, list_div


# General evaluation procedure.
def get_evaluation_results(data, model, output_length, desc):
    sum_ = [0] * output_length
    dataset_size = len(data)

    for minibatch in tqdm(data, desc):
        batch_sum = model.evaluation_step(minibatch)
        sum_ = list_add(sum_, batch_sum)

    sum_ = list_div(sum_, dataset_size)
    return {'results': sum_}
