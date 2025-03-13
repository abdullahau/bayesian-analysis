
data {
    int<lower=0> N;
    vector[N] M;
    vector[N] A;
}
parameters {
    real a;
    real bMA;
    real<lower=0> sigma;
}
transformed parameters {
    vector[N] mu;
    mu = a + bMA * M;
}
model {
    A ~ normal(mu, sigma);
    a ~ normal(0, 0.2);
    bMA ~ normal(0, 0.5);
    sigma ~ exponential(1);
}
