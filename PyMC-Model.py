import marimo

__generated_with = "0.11.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pymc as pm
    import arviz as az
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from matplotlib import style 
    style.use('PlottingStyle.mplstyle')

    print(f"Running on PyMC v{pm.__version__}")
    return az, mo, np, pd, plt, pm, style


@app.cell
def _(np):
    RANDOM_SEED = 8927
    rng = np.random.default_rng(RANDOM_SEED)
    return RANDOM_SEED, rng


@app.cell
def _(mo):
    mo.md(
        r"""
        ### A Motivating Example: Linear Regression

        $$
        \begin{aligned}
        Y  &\sim \mathcal{N}(\mu, \sigma^2) \\
        \mu &= \alpha + \beta_1 X_1 + \beta_2 X_2
        \end{aligned}
        $$

        where $\alpha$ is the intercept, and $\beta_i$ is the coefficient for covariate $X_i$, while $\sigma$ represents the observation error. Since we are constructing a Bayesian model, we must assign a prior distribution to the unknown variables in the model. We choose zero-mean normal priors with variance of 100 for both regression coefficients, which corresponds to *weak* information regarding the true parameter values. We choose a half-normal distribution (normal distribution bounded at zero) as the prior for $\sigma$.

        $$
        \begin{aligned}
        \alpha &\sim \mathcal{N}(0, 100) \\
        \beta_i &\sim \mathcal{N}(0, 100) \\
        \sigma &\sim \lvert\mathcal{N}(0, 1){\rvert}
        \end{aligned}
        $$
        """
    )
    return


@app.cell
def _(np, rng):
    # True parameter values
    alpha, sigma = 1, 1
    beta = [1, 2.5]

    # Size of dataset
    size = 100

    # Predictor variable
    X1 = np.random.randn(size)
    X2 = np.random.randn(size) * 0.2

    # Simulate outcome variable
    Y = alpha + beta[0] * X1 + beta[1] * X2 + rng.normal(size=size) * sigma
    return X1, X2, Y, alpha, beta, sigma, size


@app.cell
def _(X1, X2, Y, plt):
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
    axes[0].scatter(X1, Y, alpha=0.6)
    axes[1].scatter(X2, Y, alpha=0.6)
    axes[0].set_ylabel("Y")
    axes[0].set_xlabel("X1")
    axes[1].set_xlabel("X2");
    return axes, fig


@app.cell
def _(mo):
    mo.md(r"""### Model Specification""")
    return


@app.cell
def _(X1, X2, pm):
    def linear_model(Y):
        basic_model = pm.Model()    
        with basic_model:
            # Priors for unknown model parameters
            alpha = pm.Normal("alpha", mu=0, sigma=10)
            beta = pm.Normal("beta", mu=0, sigma=10, shape=2)
            sigma = pm.HalfNormal("sigma", sigma=1)
        
            # Expected value of outcome
            mu = alpha + beta[0] * X1 + beta[1] * X2
        
            # Likelihood (sampling distribution) of observations
            Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=Y)

            idata = pm.sample()

        return basic_model, idata
    return (linear_model,)


@app.cell
def _(Y, linear_model):
    linear_model1, idata = linear_model(Y)
    return idata, linear_model1


@app.cell
def _(idata):
    idata.posterior["alpha"].sel(draw=slice(0, 4))
    return


@app.cell
def _(mo):
    mo.md(r"""### Posterior analysis""")
    return


@app.cell
def _(az, idata):
    az.plot_trace(idata, combined=True);
    return


@app.cell
def _(az, idata):
    az.summary(idata, round_to=2, hdi_prob=0.89, kind='stats')
    return


if __name__ == "__main__":
    app.run()
