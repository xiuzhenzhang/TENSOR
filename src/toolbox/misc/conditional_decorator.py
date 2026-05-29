def conditional_decorator(*args, **kwargs):
    if len(args) + len(kwargs) == 2:
        return conditional_decorator1(*args, **kwargs)
    if len(args) + len(kwargs) == 3:
        return conditional_decorator2(*args, **kwargs)
    raise Exception(
        f'The input of conditional_decorator is either "decorator + condition" or "decorator + condition + target function". While we got {len(args) + len(kwargs)} arguments.'
    )


def conditional_decorator1(dec, condition):
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func)

    return decorator


def conditional_decorator2(dec, condition, func):
    if not condition:
        # Return the function unchanged, not decorated.
        return func
    return dec(func)
