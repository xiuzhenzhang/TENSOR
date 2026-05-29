import operator
import re

from packaging.version import Version

operator_dict = {
    '>': operator.gt,
    '<': operator.lt,
    '<=': operator.le,
    '==': operator.eq,
    '>=': operator.ge,
    '!=': operator.ne
}


def version_check(version, criteria):
    version = Version(version)
    criteria = criteria.split(',')
    print_warning = True

    for sub_criterion in criteria:
        sub_criterion = sub_criterion.strip()
        compare_label = re.match(r'^[!<>=]=?', sub_criterion).group(0)
        target_pytorch_version = Version(sub_criterion[len(compare_label):].strip())

        if not operator_dict[compare_label](version, target_pytorch_version):
            print_warning = False

    return print_warning
