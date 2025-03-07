
data {
    int<lower=0> N;
    vector[N] height;
}
parameters {
    real mu;
    real<lower=0, upper=50> sigma;
}
model {
    height ~ normal(mu, sigma);
    mu ~ normal(178, 0.1);
    sigma ~ uniform(0, 50);
}
