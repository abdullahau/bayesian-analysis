
data {
    int<lower=1> N;                     // Number of trials
    array[N] int<lower=0, upper=1> y;   // Binary outcomes
}
parameters {
    real<lower=0, upper=1> p;           // Probability of water
}
model {
    p ~ uniform(0, 1);                  // Uniform prior
    y ~ bernoulli(p);                   // Binomial likelihood
}
