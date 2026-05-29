def pack_one_value_to_dict(data, num_format = '6.5f', suffix = ''):
    return {'data': data, 'num_format': ':' + num_format, 'suffix': suffix}


def only_keep_data(dict_input):
    plain_results = {}
    for key, value in dict_input.items():
        plain_results[key] = value['data']

    return plain_results
