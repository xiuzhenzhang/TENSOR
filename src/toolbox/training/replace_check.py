def replace_check(opt, **subdirs_and_marker):
    """
    This function must ensure we use the same index in all subdirs.
    Throw an exception if indexes calculated in each dirs do not match.
    """
    if opt.replace:
        return True, "" if opt.model_index is None else str(opt.model_index)

    marker_files_exist_or_not = []

    for subdir, marker_file in subdirs_and_marker.items():
        leaf_dir_name = f"{subdir}_" + opt.model_identifier
        if isinstance(marker_file, list):
            for item in marker_file:
                tmp_path = (
                    opt.root_path / subdir / opt.procedure / str(opt.model_index) / opt.dataset_name / leaf_dir_name / item
                )
                marker_files_exist_or_not.append(tmp_path.exists())
        else:
            tmp_path = (
                opt.root_path / subdir / opt.procedure / str(opt.model_index) / opt.dataset_name / leaf_dir_name / marker_file
            )
            marker_files_exist_or_not.append(tmp_path.exists())

    if any(marker_files_exist_or_not) ^ all(marker_files_exist_or_not):
        raise ValueError(
            f"Why some marker files exist but others are not for index {opt.model_index}? This is unexpected! Perhaps we have a data loss."
        )

    return not any(marker_files_exist_or_not), str(opt.model_index)
