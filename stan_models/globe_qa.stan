
data {
    int<lower=0> W; // Number of successes (Water)
    int<lower=0> L; // Number of failures (Land)
}
parameters {
    real<lower=0, upper=1> p; // Probability of water
}
model {
    p ~ uniform(0, 1);      // Uniform prior
    W ~ binomial(W + L, p);   // Binomial likelihood
}
