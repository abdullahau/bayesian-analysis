from cmdstanpy import CmdStanModel
import numpy as np
import pandas as pd
import arviz as az
import utils as utils

from scipy import stats as stats
from matplotlib import pyplot as plt
import seaborn as sns
# ---

__all__ = ['CmdStanModel', 'np', 'pd', 'az', 'utils', 'stats', 'plt', 'sns']

# Print imports / aliases
with open(__file__) as f:
    lines = f.readlines()

from colorama import Fore
def print_import(import_line):
    if len(import_line.strip()) > 1:
        parts = [p for p in import_line.split(" ") if p]
        if parts[0] == 'import':
            module = parts[1]
            alias = parts[3]
            msg = Fore.GREEN + 'import' \
                + Fore.BLUE + f" {module} " \
                + Fore.GREEN + "as" \
                + Fore.BLUE + f" {alias}"
        elif parts[0] == 'from':
            module = parts[1]
            submodule = parts[3]
            alias = parts[5]
            msg = Fore.GREEN + 'from' \
                + Fore.BLUE + f" {module} "\
                + Fore.GREEN + 'import' \
                + Fore.BLUE + f" {submodule} " \
                + Fore.GREEN + "as" \
                + Fore.BLUE + f" {alias}"
        print(msg)

print(Fore.RED + f"Module aliases imported by init_notebook.py:\n{'-'* 44}")
for l in lines:
    if "# ---" in l:
        break
    print_import(l)

from watermark import watermark  # noqa: E402
print(Fore.RED + f"Watermark:\n{'-'* 10}")
print(Fore.BLUE + watermark())
print(Fore.BLUE + watermark(iversions=True, globals_=globals()))


import warnings  # noqa: E402
warnings.filterwarnings("ignore")

from matplotlib import style  # noqa: E402
STYLE = "PlottingStyle.mplstyle"
style.use(STYLE)
