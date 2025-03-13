
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
generated quantities {    
    // Posterior Predictive Check - y_rep (replications)
    vector[N] y_rep;
    for (i in 1:N) {
        y_rep[i] = normal_rng(mu[i], sigma);
    }
}
