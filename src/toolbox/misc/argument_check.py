def argument_check(arguments, *arguments_that_must_have, **arguments_must_be_this_class):
    if isinstance(arguments, dict):
        argument_check_dict(arguments, *arguments_that_must_have, **arguments_must_be_this_class)
    else:
        argument_check_namespace(arguments, *arguments_that_must_have, **arguments_must_be_this_class)


def argument_check_dict(arguments, *arguments_that_must_have, **arguments_must_be_this_class):
    all_arguments_that_must_have = list(set(list(arguments_that_must_have) + list(arguments_must_be_this_class.keys())))

    for item in all_arguments_that_must_have:
        try:
            arguments[item]
        except KeyError:
            raise Exception(f'Required Arguments {item} is missing!')

    for key, item in arguments_must_be_this_class.items():
        if not isinstance(arguments[key], item):
            raise TypeError(f'wrong data type at {key}! Expect: {item}, Get: {type(arguments[key])}')

    return 0


def argument_check_namespace(arguments, *arguments_that_must_have, **arguments_must_be_this_class):
    all_arguments_that_must_have = list(set(list(arguments_that_must_have) + list(arguments_must_be_this_class.keys())))

    for item in all_arguments_that_must_have:
        try:
            hasattr(arguments, item)
        except KeyError:
            raise Exception(f'Required Arguments {item} is missing!')

    for key, item in arguments_must_be_this_class.items():
        if not isinstance(getattr(arguments, key), item):
            raise TypeError(f'wrong data type at {key}! Expect: {item}, Get: {type(getattr(arguments, key))}')

    return 0

if __name__ == '__main__':
    # Test 1
    # Should error out.
    required_arguments = ['a', ]
    required_class = {'a': float, 'b': int}
    argument_dict = {'a': 1, 'b': 2}
    try:
        argument_check(argument_dict, *required_arguments, **required_class)
    except Exception as e:
        print(e)

    # Test 2
    # Should work.
    required_arguments = 'a'
    argument_dict = {'a': 1, 'b': 2}
    argument_check(argument_dict, *required_arguments)

    # Test 3
    # Should error out.
    required_arguments = ['a',]
    argument_dict = {'b': 2}
    try:
        argument_check(argument_dict, *required_arguments)
    except Exception as e:
        print(e)

    # Test 4
    # Should error out.
    required_arguments = 'a'
    argument_dict = {'b': 2}
    try:
        argument_check(argument_dict, *required_arguments)
    except Exception as e:
        print(e)

    # Test 5
    # Should work.
    required_arguments = ['a', 'b']
    argument_dict = {'a': 1, 'b': 2}
    argument_check(argument_dict, *required_arguments)


    from types import SimpleNamespace

    # Test 6
    # Should work.
    sn = SimpleNamespace()
    sn.a = 1
    sn.b = 2
    required_class = {'a': float, 'b': int}
    required_arguments = ['a', ]
    try:
        argument_check(sn, **required_class)
    except Exception as e:
        print(e)

    # Test 7
    # Should work.
    sn = SimpleNamespace()
    sn.a = 1
    sn.b = 2
    required_arguments = ['a', 'b']
    argument_check(sn, *required_arguments)

    # Test 8
    # Should error out.
    sn = SimpleNamespace()
    sn.a = 1
    sn.b = 2
    required_arguments = ['a', 'b', 'c']
    try:
        argument_check(sn, *required_arguments)
    except Exception as e:
        print(e)