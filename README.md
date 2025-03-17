# Statistical Rethinking: A Bayesian Course with Examples in Python and Stan (Second Edition)

Here is a yet another attempt to replicate (nearly) all models in Richard McElreath’s Statistical Rethinking (2nd ed.) book using Python, Stan ([CmdStanPy](https://mc-stan.org/cmdstanpy/)), [BridgeStan](https://roualdes.github.io/bridgestan/latest/), [ArviZ](https://python.arviz.org/en/stable/), and Matplotlib. *This is work in progress*.

Rendered Quarto notebooks can be viewed here:
1) [The Garden of Forking Data](html/01%20-%20The%20Garden%20of%20Forking%20Data.html)
2) [Sampling The Imaginary](html/02%20-%20Sampling%20The%20Imaginary.html)
3) [Geocentric Model (a)](html/03a%20-%20Geocentric%20Models.html)
4) [Geocentric Model (b)](html/03b%20-%20Geocentric%20Models.html)
5) [Geocentric Model (c)](html/03c%20-%20Geocentric%20Models.html)
6) [The Many Variables & The Suprious Waffles (a)](html/04a%20-%20The%20Many%20Variables%20&%20The%20Suprious%20Waffles.html)
7) [The Many Variables & The Suprious Waffles (b)](html/04b%20-%20The%20Many%20Variables%20&%20The%20Suprious%20Waffles.html)

All data and code can be downloaded from Github: https://github.com/abdullahau/bayesian-analysis

## `StanQuap`

The first part of his book, McElreath utilizes his custom `quap` function from his [rethinking](https://github.com/rmcelreath/rethinking) package. I have made an attempt to incorporate its behavior by creating a custom class [`StanQuap`](utils.py) using Stan and BridgeStan. The general outline of how `StanQuap` works is outlined below: 

- The Stan model passed as an argument is optimized using the CmdStanPy's `optimize()` API which computes the Maximum Likehood Estimate (MLE) or Maximum A Posteriori Estimate (MAP). 
- The class constructor passes the model, the data, and the unconstrained parameters from the MLE/MAP into BridgeStan's `log_density_hessian` which gives us the unconstrained hessian matrix.
- We compute the inverse of the unconstrained hessian matrix and then transform the unconstrained variance-covariance matrix into the constrained space using an analytical method (Jacobian Matrix).
- The parameter output and the variance covariance matrix can be used to samples from the posterior normal/multivariate normal distribution. 
- The methods: `laplace_sample`, `extract_samples`, `link`, and `sim` utilize CmdStanPy's `laplace_sample` API for its speed and robustness given its own parameter transformations.
