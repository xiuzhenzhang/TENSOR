# A more neat way to print hyperparameters:
def print_args(opt, opt_name):
    output = f'\n{opt_name}:'
    for key, value in opt.__dict__.items():
        output += f'\n{str(key)}: {str(value)}'

    return output
