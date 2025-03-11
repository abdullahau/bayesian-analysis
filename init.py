from cmdstanpy import CmdStanModel
import bridgestan as bs
import numpy as np
import pandas as pd
import arviz as az
import utils as utils
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
# ---

__all__ = ['CmdStanModel', 'bs', 'np', 'pd', 'az', 'utils', 'stats', 'plt', 'sns']

# Print imports / aliases
with open(__file__) as f:
    lines = f.readlines()

from colorama import Fore
import re

def print_import(import_line):
    import_line = import_line.strip()
    if not import_line or not (import_line.startswith("import") or import_line.startswith("from")):
        return  # Skip non-import lines

    # Regular expressions for parsing imports
    import_regex = re.compile(r"import\s+([\w.]+)(?:\s+as\s+(\w+))?")
    from_import_regex = re.compile(r"from\s+([\w.]+)\s+import\s+([\w.*]+)(?:\s+as\s+(\w+))?")

    # Match `import` statements
    match_import = import_regex.match(import_line)
    if match_import:
        module, alias = match_import.groups()
        msg = Fore.GREEN + "import " + Fore.BLUE + module
        if alias:
            msg += Fore.GREEN + " as " + Fore.BLUE + alias
        print(msg)
        return

    # Match `from ... import` statements
    match_from_import = from_import_regex.match(import_line)
    if match_from_import:
        module, submodule, alias = match_from_import.groups()
        msg = Fore.GREEN + "from " + Fore.BLUE + module + Fore.GREEN + " import " + Fore.BLUE + submodule
        if alias:
            msg += Fore.GREEN + " as " + Fore.BLUE + alias
        print(msg)

print(Fore.RED + f"Module aliases imported by init_notebook.py:\n{'-'*44}")
for l in lines:
    if "# ---" in l:
        break
    print_import(l)

from watermark import watermark  # noqa: E402
print()
print(Fore.RED + f"Watermark:\n{'-'* 10}")
print(Fore.BLUE + watermark())
print(Fore.BLUE + watermark(iversions=True, globals_=globals()))

import warnings  # noqa: E402
warnings.filterwarnings("ignore")

from matplotlib import style  # noqa: E402
STYLE = "PlottingStyle.mplstyle"
style.use(STYLE)
