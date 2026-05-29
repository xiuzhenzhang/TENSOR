import numbers

import numpy as np


def otd(
    pred_event_seq,
    pred_time_seq,
    true_event_seq,
    true_time_seq,
    num_events,
    add_remove_event_cost=1.0,
    move_event_cost=1.0,
    average="macro",
):
    if isinstance(add_remove_event_cost, numbers.Number):
        return otd_add_remove_event_cost_number(
            pred_event_seq,
            pred_time_seq,
            true_event_seq,
            true_time_seq,
            num_events,
            add_remove_event_cost,
            move_event_cost,
            average,
        )
    return otd_add_remove_event_cost_list(
        pred_event_seq,
        pred_time_seq,
        true_event_seq,
        true_time_seq,
        num_events,
        add_remove_event_cost,
        move_event_cost,
        average,
    )


def otd_add_remove_event_cost_list(
    pred_event_seq,
    pred_time_seq,
    true_event_seq,
    true_time_seq,
    num_events,
    add_remove_event_cost,
    move_event_cost,
    average,
):
    if pred_event_seq is None and true_event_seq is None:
        pairs, cost_of_moving = find_alignment_mc(
            pred_time_seq, true_time_seq, del_cost=add_remove_event_cost, trans_cost=move_event_cost
        )

        otds = []
        for pair, cost_of_moving_, add_remove_event_cost_ in zip(pairs, cost_of_moving, add_remove_event_cost):
            # number_of_moved_events = len(pred_time_seq) + len(true_time_seq) - 2 * len(pair)
            # otds.append((cost_of_moving_ + number_of_moved_events * add_remove_event_cost_) / len(true_time_seq))
            otds.append(cost_of_moving_ / len(true_time_seq))

        return np.stack(otds)

    assert len(pred_event_seq) == len(pred_time_seq), (
        "The predicted event sequence and time sequence have a different length. Something is definitely wrong."
    )
    assert len(true_event_seq) == len(true_time_seq), (
        "The true event sequence and time sequence have a different length. Something is definitely wrong."
    )

    assert (np.diff(pred_time_seq) >= 0).all(), (
        f"The pred_time_seq must contain absolute timestamps! Diff value: {np.diff(pred_time_seq)}."
    )
    assert (np.diff(true_time_seq) >= 0).all(), (
        f"The true_time_seq must contain absolute timestamps! Diff value: {np.diff(pred_time_seq)}."
    )

    otd_foreach_mark = [0] * num_events
    number_of_events_foreach_mark = [0] * num_events
    for mark_idx in range(num_events):
        selected_pred_time_seq = pred_time_seq[pred_event_seq == mark_idx]
        selected_true_time_seq = true_time_seq[true_event_seq == mark_idx]
        pairs, cost_of_moving = find_alignment_mc(
            selected_pred_time_seq, selected_true_time_seq, del_cost=add_remove_event_cost, trans_cost=move_event_cost
        )

        otds_for_one_mark = []
        for pair, cost_of_moving_, add_remove_event_cost_ in zip(pairs, cost_of_moving, add_remove_event_cost):
            # number_of_moved_events = len(selected_pred_time_seq) + len(selected_true_time_seq) - 2 * len(pair)
            # otds_for_one_mark.append((cost_of_moving_ + number_of_moved_events * add_remove_event_cost_))
            otds_for_one_mark.append(cost_of_moving_)

        otd_foreach_mark[mark_idx] = otds_for_one_mark
        number_of_events_foreach_mark[mark_idx] = len(selected_true_time_seq)

    if average == "macro":
        otd = np.array(otd_foreach_mark) / np.expand_dims(number_of_events_foreach_mark, -1)
        otd = np.ma.masked_invalid(otd).filled(0).mean(axis=0)
    elif average == "micro":
        otd = np.sum(otd_foreach_mark, axis=0) / np.sum(number_of_events_foreach_mark)
    elif average == "none":
        otd = np.sum(otd_foreach_mark, axis=0)
    else:
        raise Exception('Average parameter not understood. Expected values are "micro", "macro", and "none".')

    return otd


def otd_add_remove_event_cost_number(
    pred_event_seq,
    pred_time_seq,
    true_event_seq,
    true_time_seq,
    num_events,
    add_remove_event_cost,
    move_event_cost,
    average,
):
    add_remove_event_cost = [
        add_remove_event_cost,
    ]

    if pred_event_seq is None and true_event_seq is None:
        pairs, cost_of_moving = find_alignment_mc(
            pred_time_seq, true_time_seq, del_cost=add_remove_event_cost, trans_cost=move_event_cost
        )
        # number_of_moved_events = len(pred_time_seq) + len(true_time_seq) - 2 * len(pairs[0])
        # otd = (cost_of_moving[0] + number_of_moved_events * add_remove_event_cost[0]) / len(true_time_seq)
        otd = cost_of_moving[0] / len(true_time_seq)

        return otd

    assert len(pred_event_seq) == len(pred_time_seq), (
        "The predicted event sequence and time sequence have different lengths. Something is definitely wrong."
    )
    assert len(true_event_seq) == len(true_time_seq), (
        "The true event sequence and time sequence have different lengths. Something is definitely wrong."
    )

    assert (np.diff(pred_time_seq) >= 0).all(), (
        f"The pred_time_seq must contain absolute timestamps! Got: {np.diff(pred_time_seq)}"
    )
    assert (np.diff(true_time_seq) >= 0).all(), "The true_time_seq must contain absolute timestamps!"

    otd_foreach_mark = [0] * num_events
    number_of_events_foreach_mark = [0] * num_events
    for mark_idx in range(num_events):
        selected_pred_time_seq = pred_time_seq[pred_event_seq == mark_idx]
        selected_true_time_seq = true_time_seq[true_event_seq == mark_idx]
        pairs, cost_of_moving = find_alignment_mc(
            selected_pred_time_seq, selected_true_time_seq, del_cost=add_remove_event_cost, trans_cost=move_event_cost
        )
        # number_of_moved_events = len(selected_pred_time_seq) + len(selected_true_time_seq) - 2 * len(pairs[0])
        # otd_seq = cost_of_moving + number_of_moved_events * add_remove_event_cost[0]

        otd_foreach_mark[mark_idx] = cost_of_moving
        number_of_events_foreach_mark[mark_idx] = len(selected_true_time_seq)

    if average == "macro":
        otd = np.mean(
            [
                otd_one_mark / number_of_events_one_mark
                for (otd_one_mark, number_of_events_one_mark) in zip(otd_foreach_mark, number_of_events_foreach_mark)
            ]
        )
    elif average == "micro":
        otd = np.sum(otd_foreach_mark) / np.sum(number_of_events_foreach_mark)
    elif average == "none":
        otd = np.sum(otd_foreach_mark, axis=0)
    else:
        raise Exception('Average parameter not understood. Expected values are "micro", "macro", and "none".')

    return otd.item()


def find_alignment_mc(seq1, seq2, del_cost, trans_cost):
    """
    We use dynamic programming to find the best alignments between two seqs.
    ``nc'' means that this functions support a series of del_cost values.
    Note: Not support multiple types.
    :param np.ndarray seq1: Time stamps of seq #1.
    :param np.ndarray seq2: Time stamps of seq #2.
    :param np.ndarray del_cost: A series of delete cost.
    :param float trans_cost: Transportation cost per unit length.
    :return: Alignment list and minimum distances for all the del_cost values.
    """
    n_cost = len(del_cost)
    n1 = len(seq1)
    n2 = len(seq2)
    # shape=[n2, n1]
    trans_mask = np.abs(seq2.repeat(n1).reshape(n2, n1) - seq1) * trans_cost
    # shape=[n1+1, n1+1]
    del_mask = (
        np.arange(n1 + 2, dtype=np.float32)
        .repeat(n1 + 1)
        .reshape(n1 + 2, n1 + 1)
        .T.reshape(-1)[: (n1 + 1) ** 2]
        .reshape(n1 + 1, n1 + 1)
        - 1
    )
    del_mask[np.tril_indices(n1 + 1, -1)] = float("inf")
    # shape=[n1+1, n1+1, n_cost]
    del_mask = del_mask.repeat(n_cost).reshape(n1 + 1, n1 + 1, n_cost) * del_cost
    # shape=[n1+1, n1+1, n_cost]
    del_mask = del_mask.transpose([1, 0, 2]).copy()
    # shape=[n1+1, n_cost]
    overhead = np.empty(shape=[n1 + 1, n_cost], dtype=np.float32)
    overhead.fill(float("inf"))
    overhead[0, :] = 0.0
    # shape=[n2, n1+1, n_cost]
    back_pointers = np.empty(shape=[n2, n1 + 1, n_cost], dtype=np.int32)
    for n2_idx in range(n2):
        # shape=[n1+1, n1+1, n_cost]
        add_mask = del_mask.copy()
        add_mask[1:, :, :] += np.outer(
            trans_mask[n2_idx], np.ones(shape=[(n1 + 1) * n_cost], dtype=np.float32)
        ).reshape(n1, n1 + 1, n_cost)
        add_mask[np.arange(n1 + 1), np.arange(n1 + 1), :] = del_cost
        # shape=[n1+1, n1+1, n_cost]
        cost_mat = overhead + add_mask
        # shape=[n1+1, n_cost]
        choices = np.argmin(cost_mat, axis=1)
        back_pointers[n2_idx] = choices
        overhead = cost_mat.min(axis=1)
    overhead += np.outer(np.arange(n1, -1, -1, dtype=np.float32), np.ones(shape=[n_cost])) * del_cost
    # shape=[n_cost]
    curr_choice = np.argmin(overhead, axis=0)
    # shape=[n_cost]
    min_distance = overhead.min(axis=0)
    best_route = [curr_choice]
    # shape=[n1+1, n_cost]
    for choice_list in back_pointers[::-1]:
        # shape=[n_cost]
        curr_choice = choice_list[curr_choice, np.arange(n_cost)]
        best_route.append(curr_choice)
    # shape=[n2, n_cost]
    best_route = np.array(best_route)

    align_pairs = []
    for cost_idx in range(n_cost):
        best_route_ = best_route[:, cost_idx]
        pairs = []
        memo = -1
        for n2_idx_plus_1, choice_made in enumerate(best_route_[::-1]):
            if choice_made != memo:
                pairs.append([choice_made - 1, n2_idx_plus_1 - 1])
            memo = choice_made
        align_pairs.append(pairs[1:])

    return [
        align_pairs,  # len=n_cost
        min_distance,  # shape=[n_cost]
    ]


if __name__ == "__main__":

    def distance_between_event_seq(ref_seq, decode_seq, del_cost, num_types, trans_cost):
        """
        Args:
            ref_seq: [time_seqs, event_seqs]
            decode_seq: [time_seqs, event_seqs]
            del_cost:
            trans_cost:
            num_types:

        Returns:

        """
        num_cost = len(del_cost)

        distances = np.zeros(shape=[num_cost], dtype=np.float32)
        total_trans_cost = np.zeros(shape=[num_cost], dtype=np.float32)
        num_true = np.zeros(shape=[num_cost], dtype=np.int32)
        num_del = np.zeros(shape=[num_cost], dtype=np.int32)
        num_ins = np.zeros(shape=[num_cost], dtype=np.int32)
        num_align = np.zeros(shape=[num_cost], dtype=np.int32)

        seq_per_types = [[list(), list()] for _ in range(num_types)]
        for seq_idx, seq in enumerate([ref_seq, decode_seq]):
            for event_time, event_type in zip(*seq):
                if event_type >= num_types:
                    continue
                seq_per_types[event_type][seq_idx].append(event_time)

        for type_idx in range(num_types):
            ref_time = np.array(seq_per_types[type_idx][0])
            decoded_time = np.array(seq_per_types[type_idx][1])
            align_pairs, min_distance = find_alignment_mc(ref_time, decoded_time, del_cost, trans_cost)
            for cost_idx in range(num_cost):
                align_pairs_per_cost = align_pairs[cost_idx]
                min_distance_per_cost = min_distance[cost_idx]
                num_align[cost_idx] += len(align_pairs_per_cost)
                num_true[cost_idx] += len(ref_time)
                n_ins_per_cost = len(decoded_time) - len(align_pairs_per_cost)
                n_del_per_cost = len(ref_time) - len(align_pairs_per_cost)
                num_ins[cost_idx] += n_ins_per_cost
                num_del[cost_idx] += n_del_per_cost
                distances[cost_idx] += min_distance_per_cost
                total_trans_cost[cost_idx] += min_distance_per_cost - del_cost[cost_idx] * (
                    n_ins_per_cost + n_del_per_cost
                )

        return distances, total_trans_cost, num_true, num_del, num_ins, num_align

    time_seq2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    event_seq2 = np.array([0, 0, 1, 2, 3, 0, 1, 1, 2])

    time_seq1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) + 0.25
    event_seq1 = np.array([1, 0, 2, 2, 4, 4, 2, 0, 1, 4])

    otd_ = otd(
        pred_event_seq=None,
        pred_time_seq=time_seq1,
        true_event_seq=None,
        true_time_seq=time_seq2,
        num_events=5,
        add_remove_event_cost=1.0,
        move_event_cost=1.0,
        average="macro",
    )
    print(otd_)

    otd_ = otd(
        pred_event_seq=event_seq1,
        pred_time_seq=time_seq1,
        true_event_seq=event_seq2,
        true_time_seq=time_seq2,
        num_events=5,
        add_remove_event_cost=1.0,
        move_event_cost=1.0,
        average="macro",
    )
    print(otd_)

    otd_ = otd(
        pred_event_seq=event_seq1,
        pred_time_seq=time_seq1,
        true_event_seq=event_seq2,
        true_time_seq=time_seq2,
        num_events=5,
        add_remove_event_cost=1.0,
        move_event_cost=1.0,
        average="micro",
    )
    print(otd_)

    otd_ = otd(
        pred_event_seq=event_seq1,
        pred_time_seq=time_seq1,
        true_event_seq=event_seq2,
        true_time_seq=time_seq2,
        num_events=5,
        add_remove_event_cost=1.0,
        move_event_cost=1.0,
        average="none",
    )
    print(otd_)

    distances, total_trans_cost, num_true, num_del, num_ins, num_align = distance_between_event_seq(
        [time_seq2, event_seq2],
        [time_seq1, event_seq1],
        del_cost=[
            1.0,
        ],
        num_types=5,
        trans_cost=1.0,
    )
    print("Old Distance: " + str(distances))

    otd_ = otd(
        pred_event_seq=None,
        pred_time_seq=time_seq1,
        true_event_seq=None,
        true_time_seq=time_seq2,
        num_events=5,
        add_remove_event_cost=np.arange(0.1, 1.1, 0.1),
        move_event_cost=1.0,
        average="macro",
    )
    print(otd_)

    otd_ = otd(
        pred_event_seq=event_seq1,
        pred_time_seq=time_seq1,
        true_event_seq=event_seq2,
        true_time_seq=time_seq2,
        num_events=5,
        add_remove_event_cost=np.arange(0.1, 1.1, 0.1),
        move_event_cost=1.0,
        average="macro",
    )
    print(otd_)

    otd_ = otd(
        pred_event_seq=event_seq1,
        pred_time_seq=time_seq1,
        true_event_seq=event_seq2,
        true_time_seq=time_seq2,
        num_events=5,
        add_remove_event_cost=np.arange(0.1, 1.1, 0.1),
        move_event_cost=1.0,
        average="micro",
    )
    print(otd_)

    otd_ = otd(
        pred_event_seq=event_seq1,
        pred_time_seq=time_seq1,
        true_event_seq=event_seq2,
        true_time_seq=time_seq2,
        num_events=5,
        add_remove_event_cost=np.arange(0.1, 1.1, 0.1),
        move_event_cost=1.0,
        average="none",
    )
    print(otd_)

    distances, total_trans_cost, num_true, num_del, num_ins, num_align = distance_between_event_seq(
        [time_seq2, event_seq2], [time_seq1, event_seq1], del_cost=np.arange(0.1, 1.1, 0.1), num_types=5, trans_cost=1.0
    )
    print("Old Distance: " + str(distances))
