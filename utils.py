from cmdstanpy import CmdStanModel
import bridgestan as bs
import cmdstanpy as csp
import pandas as pd
import numpy as np
import arviz as az
import scipy.stats as stats
import matplotlib.pyplot as plt

import os
import gc
import json

import logging
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="plotnine/..*")
csp.utils.get_logger().setLevel(logging.ERROR)


__all__ = [
    "StanModel",
    "Stan",
    "BridgeStan",
    "StanQuap",
    "link",
    "center",
    "standardize",
    "convert_to_categorical",
    "precis",
    "precis_az",
    "cov2cor",
    "cov2corr",
    "chainmode",
    "bw_nrd0",
    "crosstab",
    "clear_memory",
]


# -------------- Load & Compile Stan Model -------------------
def StanModel(stan_file: str, stan_code: str) -> CmdStanModel:
    """Load or compile a Stan model"""
    stan_src = f"{stan_file}.stan"

    if not os.path.isfile(stan_file):
        open(stan_src, "w").write(stan_code)  # Write Stan code if needed
        return CmdStanModel(
            stan_file=stan_src,
            cpp_options={"STAN_THREADS": "true", "parallel_chains": 4},
        )

    return CmdStanModel(stan_file=stan_src, exe_file=stan_file)


class Stan(CmdStanModel):
    def __init__(self, stan_file: str, stan_code: str, force_compile=False):
        """Load or compile a Stan model"""
        stan_src = f"{stan_file}.stan"
        exe_file = stan_file

        # Check for the compiled executable
        if not os.path.isfile(exe_file) or force_compile:
            with open(stan_src, "w") as f:
                f.write(stan_code)
            super().__init__(
                stan_file=stan_src,
                force_compile=True,
                cpp_options={"STAN_THREADS": "true", "parallel_chains": 4},
            )
        else:
            super().__init__(stan_file=stan_src, exe_file=exe_file)


class BridgeStan(bs.StanModel):
    def __init__(self, stan_file: str, data: dict, force_compile=False):
        """Load or compile a BridgeStan shared object"""
        stan_so = f"{stan_file}_model.so"
        make_args = ["BRIDGESTAN_AD_HESSIAN=true", "STAN_THREADS=true"]
        data = json.dumps(data)
        if (
            not os.path.isfile(stan_so) or force_compile
        ):  # If the shared object does not exist, compile it
            super().__init__(f"{stan_file}.stan", data, make_args=make_args)
        else:
            super().__init__(stan_so, data, make_args=make_args, warn=False)


class StanQuap(object):
    """
    Description:
    Find mode of posterior distribution for arbitrary fixed effect models and
    then produce an approximation of the full posterior using the quadratic
    curvature at the mode.

    This command provides a convenient interface for finding quadratic approximations
    of posterior distributions for models defined in Stan. This procedure is equivalent
    to penalized maximum likelihood estimation and the use of a Hessian for profiling,
    and therefore can be used to define many common regularization procedures. The point
    estimates returned correspond to a maximum a posterior, or MAP, estimate. However the
    intention is that users will use `extract_samples` and `laplace_sample` and other methods to work
    with the full posterior.
    """

    def __init__(
        self,
        stan_file: str,
        stan_code: str,
        data: dict,
        algorithm="Newton",
        jacobian: bool = False,
        force_compile: bool = False,
        generated_var: list = [],
        **kwargs,
    ):
        self.train_data = data
        self.stan_model = Stan(stan_file, stan_code, force_compile)
        self.bs_model = BridgeStan(stan_file, self.train_data, force_compile)
        self.opt_model = self.stan_model.optimize(
            data=self.train_data, algorithm=algorithm, jacobian=jacobian, **kwargs
        )
        self.generated_var = generated_var
        self.params = self.opt_model.stan_variables()
        self.opt_params = {
            param: self.params[param]
            for param in self.params.keys()
            if param not in self.generated_var
        }
        self.params_unc = self.bs_model.param_unconstrain(
            np.array(self._flatten_dict_values(self.opt_params))
        )
        self.jacobian = jacobian
        self.algorithm = algorithm

    def log_density_hessian(self):
        log_dens, gradient, hessian = self.bs_model.log_density_hessian(
            self.params_unc, jacobian=self.jacobian
        )
        return log_dens, gradient, hessian

    def vcov_matrix(self, param_types=None, eps=1e-6):
        _, _, hessian_unc = self.log_density_hessian()
        vcov_unc = np.linalg.inv(-hessian_unc)
        cov_matrix = self.transform_vcov(vcov_unc, param_types, eps)
        return cov_matrix

    def laplace_sample(self, data: dict = None, draws: int = 100_000, opt_args=None):
        if data is not None:
            return self.stan_model.laplace_sample(
                data=data,
                draws=draws,
                jacobian=self.jacobian,
                opt_args=opt_args,
            )
        return self.stan_model.laplace_sample(
            data=self.train_data,
            mode=self.opt_model,
            draws=draws,
            jacobian=self.jacobian,
        )

    def extract_samples(
        self,
        n: int = 100_000,
        dict_out: bool = True,
        drop: list = None,
        select: list = None,
    ):
        if drop is None:
            drop = self.generated_var  # Default drop list

        laplace_obj = self.laplace_sample(draws=n)

        if dict_out:
            stan_var_dict = laplace_obj.stan_variables()

            # If select is provided, return only those variables
            if select is not None:
                return {
                    param: stan_var_dict[param]
                    for param in select
                    if param in stan_var_dict
                }

            # Otherwise, drop the specified variables
            return {
                param: stan_var_dict[param]
                for param in stan_var_dict.keys()
                if param not in drop
            }

        return laplace_obj.draws()

    def link(
        self,
        lm_func,
        predictor,
        n=1000,
        post=None,
        drop: list = None,
        select: list = None,
    ):
        # Extract Posterior Samples
        if post is None:
            post = self.extract_samples(n=n, dict_out=True, drop=drop, select=select)
        return lm_func(post, predictor)

    def sim(
        self, data: dict = None, n=1000, dict_out: bool = True, select: list = None
    ):
        """
        Simulate posterior observations - Posterior Predictive Sampling
        https://mc-stan.org/docs/stan-users-guide/posterior-prediction.html
        https://mc-stan.org/docs/stan-users-guide/posterior-predictive-checks.html
        """
        if select is None:
            select = self.generated_var
        if data is None:
            laplace_obj = self.laplace_sample(draws=n)
        else:
            laplace_obj = self.laplace_sample(
                data=data,
                draws=n,
                opt_args={
                    "algorithm": self.algorithm,
                    "jacobian": self.jacobian,
                },
            )
        if dict_out:
            stan_var_dict = laplace_obj.stan_variables()
            return {
                param: stan_var_dict[param]
                for param in stan_var_dict
                if param in select
            }
        return laplace_obj.draws()

    def compute_jacobian_analytical(self, param_types):
        """
        Analytical computation of the Jacobian matrix for transforming
        variance-covariance matrix from unconstrained to constrained space.
        """
        dim = len(self.params_unc)
        J = np.zeros((dim, dim))  # Initialize Jacobian matrix

        for i in range(dim):
            if param_types[i] == "uncons":  # Unconstrained (Identity transformation)
                J[i, i] = 1
            elif param_types[i] == "pos_real":  # Positive real (Exp transformation)
                J[i, i] = np.exp(self.params_unc[i])
            elif param_types[i] == "prob":  # Probability (Logit transformation)
                x = 1 / (1 + np.exp(-self.params_unc[i]))  # Sigmoid function
                J[i, i] = x * (1 - x)
            else:
                raise ValueError(f"Unknown parameter type: {param_types[i]}")

        return J

    def compute_jacobian_numerical(self, eps=1e-6):
        """
        Analytical computation of the Jacobian matrix for transforming
        variance-covariance matrix from unconstrained to constrained space.
        """
        dim = len(self.params_unc)
        J = np.zeros((dim, dim))  # Full Jacobian matrix

        # Compute Jacobian numerically for each parameter
        for i in range(dim):
            perturbed = self.params_unc.copy()
            # Perturb parameter i
            perturbed[i] += eps
            constrained_plus = np.array(self.bs_model.param_constrain(perturbed))
            perturbed[i] -= 2 * eps
            constrained_minus = np.array(self.bs_model.param_constrain(perturbed))
            # Compute numerical derivative
            J[:, i] = (constrained_plus - constrained_minus) / (2 * eps)

        return J

    def transform_vcov(self, vcov_unc, param_types=None, eps=1e-6):
        """
        Transform the variance-covariance matrix from the unconstrained space to the constrained space.
        Args:
        - vcov_unc (np.array): variance-covariance matrix in the unconstrained space.
        - param_types (list) [Required for analytical solution]: List of strings specifying the type of each parameter.
        Options: 'uncons' (unconstrained), 'pos_real' (positive real), 'prob' (0 to 1).
        - eps (float) [Required for numerical solution]: Small perturbation for numerical differentiation.
        Returns:
        - vcov_con (np.array): variance-covariance matrix in the constrained space.
        """
        if param_types is None:
            J = self.compute_jacobian_numerical(eps)
        else:
            J = self.compute_jacobian_analytical(param_types)
        vcov_con = J.T @ vcov_unc @ J
        return vcov_con

    def precis(self, param_types=None, prob=0.89, eps=1e-6):
        vcov_mat = self.vcov_matrix(param_types, eps)
        pos_mu = np.array(self._flatten_dict_values(self.opt_params))
        pos_sigma = np.sqrt(np.diag(vcov_mat))
        plo = (1 - prob) / 2
        phi = 1 - plo
        lo = pos_mu + pos_sigma * stats.norm.ppf(plo)
        hi = pos_mu + pos_sigma * stats.norm.ppf(phi)
        res = pd.DataFrame(
            {
                "Parameter": self.bs_model.param_names(),
                "Mean": pos_mu,
                "StDev": pos_sigma,
                f"{plo:.1%}": lo,
                f"{phi:.1%}": hi,
            }
        )
        return res.set_index("Parameter")

    def _flatten_dict_values(self, d):
        arrays = [np.ravel(np.array(value)) for value in d.values()]
        return np.concatenate(arrays)


def link(fit, lm_func, predictor, n=1000, post=None):
    # Extract Posterior Samples
    if post is None:
        post = fit.extract_samples(n=n, dict_out=True)
    return lm_func(post, predictor)


# ----------------------- Stat Functions -----------------------
def center(vals: np.ndarray) -> np.ndarray:
    return vals - np.nanmean(vals)


def standardize(vals: np.ndarray, ddof=1) -> np.ndarray:
    centered_vals = center(vals)
    return centered_vals / np.nanstd(vals, ddof=ddof)


def convert_to_categorical(vals):
    return vals.astype("category").cat.codes.values


def logit(p: float) -> float:
    return np.log(p / (1 - p))


def invlogit(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def precis(samples, prob=0.89, index_name="Parameter", np_axis=0):
    if isinstance(samples, dict) or isinstance(samples, pd.Series):
        samples = pd.DataFrame(samples)

    plo = (1 - prob) / 2
    phi = 1 - plo

    if isinstance(samples, np.ndarray):
        return (
            samples.mean(axis=np_axis),
            np.quantile(samples, q=plo, axis=np_axis),
            np.quantile(samples, q=phi, axis=np_axis),
        )

    res = pd.DataFrame(
        {
            f"{index_name}": samples.columns.to_numpy(),
            "Mean": samples.mean().to_numpy(),
            "StDev": samples.std().to_numpy(),
            f"{plo:.1%}": samples.quantile(q=plo).to_numpy(),
            f"{phi:.1%}": samples.quantile(q=phi).to_numpy(),
        }
    )
    return res.set_index(f"{index_name}")


def precis_az(samples, var_names=None):
    return az.summary(samples, kind="stats", hdi_prob=0.89, var_names=var_names).round(
        2
    )


def cov2cor(c: np.ndarray) -> np.ndarray:
    """
    Return a correlation matrix given a covariance matrix.
    : c = covariance matrix
    """
    D = np.zeros(c.shape)
    np.fill_diagonal(D, np.sqrt(np.diag(c)))
    invD = np.linalg.inv(D)
    return invD @ c @ invD


# finds mode of a continuous density
def chainmode(chain, bw_fct=0.01, **kwargs):
    x, y = az.kde(chain, bw_fct=bw_fct, **kwargs)
    return x[np.argmax(y)]


def cov2corr(A):
    """
    covariance matrix to correlation matrix.
    https://www.r-bloggers.com/2019/08/arguments-of-statsdensity/
    https://python.arviz.org/en/latest/api/generated/arviz.kde.html
    https://python.arviz.org/en/stable/_modules/arviz/stats/density_utils.html#kde
    """
    d = np.sqrt(A.diagonal())  # Compute standard deviations
    A = ((A.T / d).T) / d  # # Normalize each element by sqrt(C_ii * C_jj)
    return A


# R's bandwidth: Bandwidth Selectors for Kernel Density Estimation
# https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/bandwidth
# Other Bandwidth selections include: bw.nrd, bw.ucv, bw.bcv, and bw.SJ


def bw_nrd0(x):
    """
    Implementation of R's rule-of-thumb for choosing the bandwidth of a Gaussian
    kernel density estimator. It defaults to 0.9 times the minimum of the standard
    deviation and the interquartile range divided by 1.34 times the sample size to
    the negative one-fifth power (= Silverman's ‘rule of thumb’, Silverman (1986,
    page 48, eqn (3.31))) unless the quartiles coincide when a positive result
    will be guaranteed.
    """
    if len(x) < 2:
        raise (Exception("need at least 2 data points"))

    hi = np.std(x, ddof=1)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    lo = min(hi, iqr / 1.34)

    lo = lo or hi or abs(x[0]) or 1

    # if not lo:
    #     if hi:
    #         lo = hi
    #     elif abs(x[0]):
    #         lo = abs(x[0])
    #     else:
    #         lo = 1

    return 0.9 * lo * len(x) ** -0.2


# bw.nrd is the more common variation given by Scott (1992), using factor 1.06.

# bw.ucv and bw.bcv implement unbiased and biased cross-validation respectively.

# bw.SJ implements the methods of Sheather & Jones (1991) to select the bandwidth
# using pilot estimation of derivatives. The algorithm for method "ste" solves an equation
# (via uniroot) and because of that, enlarges the interval c(lower, upper) when the boundaries
# were not user-specified and do not bracket the root.


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
    default_exceptions = ["exceptions", "active_variables"]
    if exceptions:
        default_exceptions.extend(exceptions)  # Append user-provided exceptions

    # Default targeted types
    default_types = [
        dict,
        list,
        int,
        float,
        str,
        tuple,
        CmdStanModel,
        plt.Axes,
        az.InferenceData,
        pd.DataFrame,
        pd.Series,
        plt.Figure,
        np.ndarray,
        np.int64,
        np.float32,
    ]

    if targeted_types:
        default_types.extend(targeted_types)  # Append user-provided types

    # Identify variables to delete
    active_variables = [
        var
        for var, value in globals().items()
        if not var.startswith("_")  # Exclude private/internal variables
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
