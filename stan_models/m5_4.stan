
data {
    int<lower=0> N;
    vector[N] M;
    vector[N] A;
}
parameters {
    real a;
    real bAM;
    real<lower=0> sigma;
}
transformed parameters {
    vector[N] mu;
    mu = a + bAM * A;
}
model {
    M ~ normal(mu, sigma);
    a ~ normal(0, 0.2);
    bAM ~ normal(0, 0.5);
    sigma ~ exponential(1);
}
