def list_to_string(*seqs):
    if len(seqs) == 1:
        return ' '.join(map(str, seqs[0]))

    result = []
    for seq in seqs:
        seq_string = ' '.join(map(str, seq))
        result.append(seq_string)

    return result
