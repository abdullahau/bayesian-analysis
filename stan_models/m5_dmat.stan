
data {
    int<lower=1> N;   // number of observations
    int<lower=1> K;   // number of regressors (including constant)
    vector[N] D;      // outcome
    matrix[N, K] X;   // regressors
}
parameters {
    real<lower=0> sigma;    // scale
    vector[K] b;            // coefficients (including constant)
}
transformed parameters {
    vector[N] mu = X * b;   // location
}
model {
    D ~ normal(mu, sigma);    // probability model
    sigma ~ exponential(1);   // prior for scale
    b[1] ~ normal(0, 0.2);    // prior for intercept
    for (i in 2:K) {          // priors for coefficients
        b[i] ~ normal(0, 0.5);
    }
}
generated quantities {
    vector[N] y_tilde;           // predicted outcome
    for (i in 1:N) y_tilde[i] = normal_rng(mu[i], sigma);
}
