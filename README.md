# Statistical Rethinking: A Bayesian Course with Examples in Python and Stan (Second Edition)

Here is yet another attempt to replicate (nearly) all models in Richard McElreath’s *Statistical Rethinking (2nd ed.)* book using Python, Stan ([CmdStanPy](https://mc-stan.org/cmdstanpy/)), [BridgeStan](https://roualdes.github.io/bridgestan/latest/), and [ArviZ](https://python.arviz.org/en/stable/). *This is a work in progress.*

Rendered Quarto notebooks can be viewed here:

1)  [The Garden of Forking Data](https://abdullahau.github.io/bayesian-analysis/01%20-%20The%20Garden%20of%20Forking%20Data)
2)  [Sampling The Imaginary](https://abdullahau.github.io/bayesian-analysis/02%20-%20Sampling%20The%20Imaginary)
3)  [(a) Geocentric Model](https://abdullahau.github.io/bayesian-analysis/03a%20-%20Geocentric%20Models)
4)  [(b) Geocentric Model](https://abdullahau.github.io/bayesian-analysis/03b%20-%20Geocentric%20Models)
5)  [(c) Geocentric Model](https://abdullahau.github.io/bayesian-analysis/03c%20-%20Geocentric%20Models)
6)  [(a) The Many Variables & The Spurious Waffles](https://abdullahau.github.io/bayesian-analysis/04a%20-%20The%20Many%20Variables%20&%20The%20Suprious%20Waffles)
7)  [(b) The Many Variables & The Spurious Waffles](https://abdullahau.github.io/bayesian-analysis/04b%20-%20The%20Many%20Variables%20&%20The%20Suprious%20Waffles)

All data and code can be downloaded from GitHub: https://github.com/abdullahau/bayesian-analysis

---

# `StanQuap`

## Overview

In the first part of his book, Richard McElreath utilizes his custom `quap` function from the [rethinking](https://github.com/rmcelreath/rethinking) package. To replicate its behavior using Stan and BridgeStan, I created a custom class called `StanQuap`. This class approximates the full posterior distribution by leveraging quadratic curvature at the mode.

## How `StanQuap` Works

- The provided Stan model is optimized using CmdStanPy's `optimize()` API to compute the Maximum Likelihood Estimate (MLE) or Maximum A Posteriori Estimate (MAP).
- The class constructor passes the model, data, and unconstrained parameters from the MLE/MAP into BridgeStan's `log_density_hessian`, which computes (or provides access to) the unconstrained Hessian matrix.
- The inverse of the unconstrained Hessian matrix is computed and transformed into the constrained space using an analytical method (Jacobian matrix).
- The posterior distribution is approximated as a normal/multivariate normal distribution, where the mean is the parameter output and the covariance is the variance-covariance matrix.
- The methods `laplace_sample`, `extract_samples`, `link`, and `sim` utilize CmdStanPy's `laplace_sample` API for speed and robustness, ensuring proper parameter transformations.

## Features

- **Mode Finding**: Computes the mode of the posterior distribution.
- **Variance-Covariance Approximation**: Estimates uncertainty using the Hessian.
- **Posterior Sampling**: Draws from the posterior distribution via the Laplace approximation.
- **Link Function**: Transforms posterior samples using a user-defined function.
- **Posterior Prediction**: Simulates posterior observations for predictive checks.
- **Jacobian Transformation**: Converts unconstrained variance-covariance matrices into constrained space.

## Usage

### Example

```python
import utils

bernoulli = """
data {
  int<lower=1> N;
  array[N] int<lower=0,upper=1> y;
}
parameters {
  real<lower=0,upper=1> theta;
}
model {
  theta ~ beta(1,1);
  y ~ bernoulli(theta);
}
"""

data = {"N":10,"y":[0,1,0,1,0,0,0,0,0,1]}

# Define StanQuap Model
quap = utils.StanQuap(
    stan_file="bernoulli_model",
    stan_code=bernoulli,
    data=data,
)

# Extract Samples
samples = quap.extract_samples(n=1000)

# Compute Posterior Summary
summary = quap.precis()
print(summary)
```

## Dependencies

- [CmdStanPy](https://github.com/stan-dev/cmdstanpy)
- [BridgeStan](https://github.com/roualdes/bridgestan)
- [ArviZ](https://github.com/arviz-devs/arviz)
- NumPy, SciPy, Pandas, Matplotlib, Seaborn, Formulaic, Daft-PGM

## Installation

Ensure you have CmdStan and BridgeStan installed:

```bash
pip install -r requirements.txt
```

Follow the [CmdStanPy installation guide](https://mc-stan.org/cmdstanpy/installation.html) and [BridgeStan installation guide](https://github.com/roualdes/bridgestan) for additional setup.

## License

This project is licensed under the MIT License.
