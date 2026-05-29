def dump_to_pkl(data, filepath, compression = None):
    import importlib
    import os
    import pickle as pkl
    from pathlib import Path

    dict_compression_algorithms = {
        None: open,
        # Is it a good choice?
        'lzma': importlib.import_module('lzma').open,
        'bz2': importlib.import_module('bz2').open,
        'gz': importlib.import_module('gzip').open
    }
    '''
    Add proper suffix to the base file name if compression is not None.
    '''
    head, tail = os.path.split(filepath)
    tail = tail + f'{"." + compression if compression is not None else ""}'
    filepath = Path(head, tail)

    selected_open_function = dict_compression_algorithms[compression]
    f = selected_open_function(filepath, 'wb')
    pkl.dump(data, f)
    f.close()

    return 0


def dump_to_npz(filepath, **kwargs):
    import numpy as np

    '''
    Add proper suffix to the base file name if compression is not None.
    '''
    np.savez(filepath, **kwargs)

    return 0


def write_to_txt(strings, filepath):
    from pathlib import Path

    f = Path(filepath).open("w")

    if isinstance(strings, list):
        strings = [item + '\n' for item in strings]
        f.writelines(strings)
    else:
        f.write(strings)

    f.close()

    return 0
