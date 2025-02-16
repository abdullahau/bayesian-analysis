import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from matplotlib import pyplot as plt

def crosstab(x: np.array, y: np.array, labels: list[str] = None):
    """Simple cross tabulation of two discrete vectors x and y"""
    ct = pd.crosstab(x, y)
    if labels:
        ct.index = labels
        ct.columns = labels
    return ct


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


def simulate_2_parameter_bayesian_learning_grid_approximation(
    x_obs,
    y_obs,
    param_a_grid,
    param_b_grid,
    true_param_a,
    true_param_b,
    model_func,
    posterior_func,
    n_posterior_samples=3,
    param_labels=None,
    data_range_x=None,
    data_range_y=None,
):
    """General function for simulating Bayesian learning in a 2-parameter model
    using grid approximation.

    Parameters
    ----------
    x_obs : np.ndarray
        The observed x values
    y_obs : np.ndarray
        The observed y values
    param_a_grid: np.ndarray
        The range of values the first model parameter in the model can take.
        Note: should have same length as param_b_grid.
    param_b_grid: np.ndarray
        The range of values the second model parameter in the model can take.
        Note: should have same length as param_a_grid.
    true_param_a: float
        The true value of the first model parameter, used for visualizing ground
        truth
    true_param_b: float
        The true value of the second model parameter, used for visualizing ground
        truth
    model_func: Callable
        A function `f` of the form `f(x, param_a, param_b)`. Evaluates the model
        given at data points x, given the current state of parameters, `param_a`
        and `param_b`. Returns a scalar output for the `y` associated with input
        `x`.
    posterior_func: Callable
        A function `f` of the form `f(x_obs, y_obs, param_grid_a, param_grid_b)
        that returns the posterior probability given the observed data and the
        range of parameters defined by `param_grid_a` and `param_grid_b`.
    n_posterior_samples: int
        The number of model functions sampled from the 2D posterior
    param_labels: Optional[list[str, str]]
        For visualization, the names of `param_a` and `param_b`, respectively
    data_range_x: Optional len-2 float sequence
        For visualization, the upper and lower bounds of the domain used for model
        evaluation
    data_range_y: Optional len-2 float sequence
        For visualization, the upper and lower bounds of the range used for model
        evaluation.
    """
    param_labels = param_labels if param_labels is not None else ["param_a", "param_b"]
    data_range_x = (x_obs.min(), x_obs.max()) if data_range_x is None else data_range_x
    data_range_y = (y_obs.min(), y_obs.max()) if data_range_y is None else data_range_y

    # NOTE: assume square parameter grid
    resolution = len(param_a_grid)

    param_a_grid, param_b_grid = np.meshgrid(param_a_grid, param_b_grid)
    param_a_grid = param_a_grid.ravel()
    param_b_grid = param_b_grid.ravel()

    posterior = posterior_func(x_obs, y_obs, param_a_grid, param_b_grid)

    # Visualization
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot Posterior over intercept and slope params
    plt.sca(axs[0])
    plt.contour(
        param_a_grid.reshape(resolution, resolution),
        param_b_grid.reshape(resolution, resolution),
        posterior.reshape(resolution, resolution),
        cmap="gray_r",
    )

    # Sample locations in parameter space according to posterior
    sample_idx = np.random.choice(
        np.arange(len(posterior)),
        p=posterior / posterior.sum(),
        size=n_posterior_samples,
    )

    param_a_list = []
    param_b_list = []
    for ii, idx in enumerate(sample_idx):
        param_a = param_a_grid[idx]
        param_b = param_b_grid[idx]
        param_a_list.append(param_a)
        param_b_list.append(param_b)

        # Add sampled parameters to posterior
        plt.scatter(param_a, param_b, s=60, c=f"C{ii}", alpha=0.75, zorder=20)

    # Add the true params to the plot for reference
    plt.scatter(
        true_param_a, true_param_b, color="k", marker="x", s=60, label="true parameters"
    )

    plt.xlabel(param_labels[0])
    plt.ylabel(param_labels[1])

    # Plot the current training data and model trends sampled from posterior
    plt.sca(axs[1])
    plt.scatter(x_obs, y_obs, s=60, c="k", alpha=0.5)

    # Plot the resulting model functions sampled from posterior
    xs = np.linspace(data_range_x[0], data_range_x[1], 100)
    for ii, (param_a, param_b) in enumerate(zip(param_a_list, param_b_list)):
        ys = model_func(xs, param_a, param_b)
        plt.plot(xs, ys, color=f"C{ii}", linewidth=4, alpha=0.5)

    groundtruth_ys = model_func(xs, true_param_a, true_param_b)
    plt.plot(
        xs, groundtruth_ys, color="k", linestyle="--", alpha=0.5, label="true trend"
    )

    plt.xlim([data_range_x[0], data_range_x[1]])
    plt.xlabel("x value")

    plt.ylim([data_range_y[0], data_range_y[1]])
    plt.ylabel("y value")

    plt.title(f"N={len(y_obs)}")
    plt.legend(loc="upper left")

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

def precis(samples, var_names=None):
    return az.summary(samples, kind="stats", hdi_prob=0.89, var_names=var_names).round(2)

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

# See https://python.arviz.org/en/stable/getting_started/CreatingInferenceData.html#from-dataframe
# def convert_to_inference_data(df):
#     df["chain"] = 1
#     df["draw"] = np.arange(len(df), dtype=int)
#     df = df.set_index(["chain", "draw"])
#     xdata = xr.Dataset.from_dataframe(df)
#     idata = az.InferenceData(posterior=xdata)
#     return idata

# # Add the `to_inference_data()` method to the DataFrame class
# pd.DataFrame.convert_to_inference_data = convert_to_inference_data

# def extract_samples(custom_step, size=10000):
#     samples = np.random.multivariate_normal(mean=custom_step.mode, cov=custom_step.covariance, size=size)
#     df = pd.DataFrame({"mu": samples[:, 0], "sigma": samples[:, 1]})
#     return df.convert_to_inference_data()