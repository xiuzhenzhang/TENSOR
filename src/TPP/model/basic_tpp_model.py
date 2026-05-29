from abc import ABCMeta, abstractmethod

import torch.nn as nn

# We use this parameter to control model's memory usage while running event-time prediction tasks.
# Used in FENN, FullyNN, SAHP, RHP, and THP
memory_ceiling = 3e8

# The lower and upper boundary of the inversed transform sampling.
# The final trick to make IFIB generate sane samples by avoiding the long tail.
its_lower_bound = 0.0
its_upper_bound = 0.9


class BasicModel(nn.Module, metaclass=ABCMeta):
    """
    The parent of all model classes.
    """
    @abstractmethod
    def forward(self, *args):
        """
        The entry function of all model. Pytorch can automatically move the data batch to correct device
        because we pack all model by DistributedParallel(DP) or DistributedDataParallel(DDP), .
        However, this feature only works when you access the model through forward().
        """
        return NotImplementedError("Please Implement forward()!")

    @abstractmethod
    def next_event_prediction_time_mark(self, *args):
        """
        """
        return NotImplementedError("next_event_prediction_time_mark() not implemented!")

    # @abstractmethod
    # def next_one_event_prediction_time_mark(self, *args):
    #     """
    #     """
    #     return NotImplementedError("next_event_prediction_time_mark() not implemented!")

    @abstractmethod
    def next_event_prediction_mark_time(self, *args):
        """
        """
        return NotImplementedError("meannext_event_prediction_mark_time_absolute_error_e() not implemented!")

    # @abstractmethod
    # def next_one_event_prediction_mark_time(self, *args):
    #     """
    #     """
    #     return NotImplementedError("meannext_event_prediction_mark_time_absolute_error_e() not implemented!")

    @abstractmethod
    def probability_time_next_2d(self, *args):
        """
        """
        return NotImplementedError("meannext_event_prediction_mark_time_absolute_error_e() not implemented!")

    @abstractmethod
    def train_step(self, minibatch):
        """
        Please tell us how your model propagates and obtains a proper loss value using one minibatch from the training dataset.
        """
        return NotImplementedError("Please Implement train_step()!")

    @abstractmethod
    def evaluation_step(self, minibatch):
        """
        Please tell us how your model propagates and obtains a proper loss value using one minibatch from the evaluation dataset.
        """
        return NotImplementedError("Please Implement evaluation_step()!")

    @abstractmethod
    def postprocess(self, input_data, procedure):
        """
        You can do whatever postprocess here on the raw results from train_step() and evaluation_step().
        The input is the output of function train_step() or function evaluation_step(). You should return a list.
        """
        return input_data

    """
    The input of log_print_format() and logfile_print_format() is the output object of function postprocess()
    """

    @abstractmethod
    def log_print_format(self, input_data):
        """
        The output format definition. The rule-defining dict should contain objects listed below:
        1. 'num_format': Please, do not modify the name because the architecture will detect this key and use the corresponding subdict as the output format definition.
        2. What you want to output. You should register the name of each number in list 'input' as a key and each matching number as a value.
        Caveats: All used names should have their own format definition. If you really don't need it for some special outputs, please set it to an empty string ''.
        e.x.:
        input = [a, b]. Expected output: loss_a: a, loss_b: b. Both a and b should keep 5 decimal places.
        The format_dict should be like this:
        {
            'loss_a': a,
            'loss_b': b,
            'num_format': {'loss_a': ':.5f', 'relative_loss': ':.5f'}
        }
        """

    # The largest length of the format_dict
    format_dict_length = 0

    metric_number = 0  # metric number is the length of the output of choose_metric
    """
    evaluation_report and test_report have the same variable mapping with postprocess.
    """

    @abstractmethod
    def choose_metric(self, evaluation_report, test_report):
        """
        Choose the metric values that you want to employ for model performance comparison.

        You'd better to mark the name of each object in the output list as a reminder, like:
        [relative loss on evaluation dataset, relative loss on test dataset]
        """
        return NotImplementedError("please tell us what indicates a better checkpoint.")
