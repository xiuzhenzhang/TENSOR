def flatten(np_lists):
    results = []
    for np_list in np_lists:
        results += np_list.flatten().tolist()
    return results
