# How to print formated logs via logger and format definitions.
def print_performances(logger, procedure, data_dict):
    info_string = ""
    for key, value in data_dict.items():
        sub_info = " ," + key + ": {" + value["num_format"] + "}" + value["suffix"]
        info_string += sub_info.format(value["data"])

    info_string = f"{procedure:12}" + info_string
    logger.info(info_string)
