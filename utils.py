from cmdstanpy import CmdStanModel, CmdStanMCMC
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

import os
import gc

# -------------- Load & Compile Stan Model -------------------
def StanModel(stan_file: str, stan_code: str) -> CmdStanModel:
    """Load or compile a Stan model"""
    stan_src = f"{stan_file}.stan"

    if not os.path.isfile(stan_file):  
        open(stan_src, 'w').write(stan_code)  # Write Stan code if needed
        return CmdStanModel(stan_file=stan_src, cpp_options={'STAN_THREADS': 'true', 'parallel_chains': 4})
    
    return CmdStanModel(stan_file=stan_src, exe_file=stan_file)

class Stan(CmdStanModel):
    def __init__(self, stan_file: str, stan_code: str):
        """Load or compile a Stan model"""
        stan_src = f"{stan_file}.stan"
        exe_file = stan_file
        
        # Check for the compiled executable
        if not os.path.isfile(exe_file):
            with open(stan_src, 'w') as f:
                f.write(stan_code)
            super().__init__(stan_file=stan_src, cpp_options={'STAN_THREADS': 'true', 'parallel_chains': 4})
        else:
            super().__init__(stan_file=stan_src, exe_file=exe_file)

class BridgeStan(bs.StanModel):
    def __init__(self, stan_file: str, data: dict):
        """Load or compile a BridgeStan shared object"""
        stan_so = f"{stan_file}_model.so"
        make_args = ['BRIDGESTAN_AD_HESSIAN=true', 'STAN_THREADS=true']
        data = json.dumps(data)
        if not os.path.isfile(stan_so):  # If the shared object does not exist, compile it
            super().__init__(f"{stan_file}.stan", data, make_args=make_args)
        else:
            super().__init__(stan_so, data, make_args=make_args)

def quap_precis(model: Stan, data: dict, jacobian=False,**kwargs):
    stan_file = model.exe_file
    opt_model = model.optimize(data, algorithm='BFGS', jacobian=jacobian, **kwargs)
    bs_model = BridgeStan(stan_file, data)
    params = opt_model.stan_variables()
    mode_params_unc = bs_model.param_unconstrain(
        np.array(list(params.values()))  
    )
    log_dens, gradient, hessian = bs_model.log_density_hessian(mode_params_unc, jacobian=jacobian)
    cov_matrix = np.linalg.inv(-hessian)
    
    return opt_model, params, cov_matrix, hessian

# ----------------------- Stat Functions -----------------------
def center(vals: np.ndarray) -> np.ndarray:
    return vals - np.nanmean(vals)


def standardize(vals: np.ndarray) -> np.ndarray:
    centered_vals = center(vals)
    return centered_vals / np.nanstd(centered_vals)


def convert_to_categorical(vals):
    return vals.astype("category").cat.codes.values


def logit(p: float) -> float:
    return np.log(p / (1 - p))


def invlogit(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def precis(samples, var_names=None):
    return az.summary(samples, kind="stats", hdi_prob=0.89, var_names=var_names).round(2)

    
def vcov(custom_step):
    '''Returns Variance-Covariance matrix of the parameters
    Input:
            custom_step: pymc.step_methods.arraystep
    '''
    return custom_step.covariance

def cov2cor(c: np.ndarray) -> np.ndarray:
    """
    Return a correlation matrix given a covariance matrix.
    : c = covariance matrix
    """
    D = np.zeros(c.shape)
    np.fill_diagonal(D, np.sqrt(np.diag(c)))
    invD = np.linalg.inv(D)
    return invD @ c @ invD

# R's bandwidth: Bandwidth Selectors for Kernel Density Estimation
# https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/bandwidth
# Other Bandwidth selections include: bw.nrd, bw.ucv, bw.bcv, and bw.SJ

def bw_nrd0(x):
    '''
    Implementation of R's rule-of-thumb for choosing the bandwidth of a Gaussian 
    kernel density estimator. It defaults to 0.9 times the minimum of the standard 
    deviation and the interquartile range divided by 1.34 times the sample size to 
    the negative one-fifth power (= Silverman's ‘rule of thumb’, Silverman (1986, 
    page 48, eqn (3.31))) unless the quartiles coincide when a positive result 
    will be guaranteed.
    '''
    if len(x) < 2:
        raise(Exception("need at least 2 data points"))

    hi = np.std(x, ddof=1)
    q75, q25 = np.percentile(x, [75 ,25])
    iqr = q75 - q25
    lo = min(hi, iqr/1.34)
    
    lo = lo or hi or abs(x[0]) or 1

    # if not lo:
    #     if hi:
    #         lo = hi
    #     elif abs(x[0]):
    #         lo = abs(x[0])
    #     else:
    #         lo = 1

    return 0.9 * lo *len(x)**-0.2

# bw.nrd is the more common variation given by Scott (1992), using factor 1.06.

# bw.ucv and bw.bcv implement unbiased and biased cross-validation respectively.

# bw.SJ implements the methods of Sheather & Jones (1991) to select the bandwidth 
# using pilot estimation of derivatives. The algorithm for method "ste" solves an equation 
# (via uniroot) and because of that, enlarges the interval c(lower, upper) when the boundaries
# were not user-specified and do not bracket the root.

from pymc.step_methods.arraystep import ArrayStep
from pymc.util import get_value_vars_from_user_vars

class QuadraticApproximation(ArrayStep):
    def __init__(self, vars, model, start=None, **kwargs):
        self.model = model
        self.vars = vars
        self.varnames = [var.name for var in vars]

        self.mode, self.covariance = self._compute_mode_and_covariance(start)

        vars = get_value_vars_from_user_vars(vars, model)

        super().__init__(vars, [self._logp_fn], **kwargs)

    def _point_to_array(self, point):
        return np.array([point[varname] for varname in self.varnames])

    def _array_to_point(self, array):
        return {varname: val for varname, val in zip(self.varnames, array)}

    def _logp_fn(self, x):
        point = self._array_to_point(x)
        return self.model.logp(point)

    def _compute_mode_and_covariance(self, start=None):

        map = pm.find_MAP(vars=self.vars, start=start, progressbar=False)

        m = pm.modelcontext(None)

        for var in self.vars:
            if m.rvs_to_transforms[var] is not None:
                m.rvs_to_transforms[var] = None
                var_value = m.rvs_to_values[var]
                var_value.name = var.name

        H = pm.find_hessian(map, vars=self.vars)
        cov = np.linalg.inv(H)
        mean = np.concatenate([np.atleast_1d(map[v.name]) for v in self.vars])

        return mean, cov

    def astep(self, q0, logp):
        sample = np.random.multivariate_normal(self.mode, self.covariance)
        return sample, []

# ----------------- Quarto Inline Plotting -----------------
def inline_plot(plot_func, *args, **kwargs):
    """
    A helper function to display plots inline in Quarto (RStudio).

    Parameters:
    - plot_func: Function that generates the plot (e.g., plt.plot).
    - *args, **kwargs: Arguments to pass to the plot function.
    """
    plt.clf()  # Clear any existing plot
    plot_func(*args, **kwargs)  # Call the plotting function with arguments
    plt.show()  # Show the plot inline
    plt.close()  # Close the plot to avoid display issues

# Example Use:
# def my_custom_plot():
#     plt.plot([1, 2, 3, 4], [10, 20, 25, 30], marker='o')
#     plt.title("My Custom Plot")
# 
# inline_plot(my_custom_plot)

# ----------------------- Crosstable -----------------------
def crosstab(x: np.array, y: np.array, labels: list[str] = None):
    """Simple cross tabulation of two discrete vectors x and y"""
    ct = pd.crosstab(x, y)
    if labels:
        ct.index = labels
        ct.columns = labels
    return ct

# ----------------------- Garbage Collect -----------------------
def clear_memory(exceptions=None, targeted_types=None):
    """Clears memory by deleting global variables except those in exceptions.
    
    Args:
        exceptions (list, optional): List of variable names to exclude from deletion.
        targeted_types (list, optional): List of data types to delete.
    """

    # Default exceptions
    default_exceptions = ['exceptions', 'active_variables']
    if exceptions:
        default_exceptions.extend(exceptions)  # Append user-provided exceptions

    # Default targeted types
    default_types = [CmdStanModel, CmdStanMCMC, plt.Axes, az.InferenceData, 
                     pd.DataFrame, pd.Series, dict, list, int, float, str, 
                     tuple, plt.Figure, defaultdict, np.ndarray, np.int64, 
                     np.float32]
    if targeted_types:
        default_types.extend(targeted_types)  # Append user-provided types

    # Identify variables to delete
    active_variables = [
        var for var, value in globals().items()
        if not var.startswith('_')  # Exclude private/internal variables
        and var not in default_exceptions  # Exclude user-specified exceptions
        and isinstance(value, tuple(default_types))  # Check against expanded type list
    ]

    # Delete identified variables
    for var in active_variables:
        del globals()[var]

    # Cleanup references
    del active_variables, default_exceptions, default_types

    # Run garbage collection
    gc.collect()

