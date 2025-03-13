
data {
    int<lower=0> N;
    vector[N] M;
    vector[N] D;
    vector[N] A;
}
parameters {
    real a;
    real bA;
    real bM;
    real<lower=0> sigma;
}
transformed parameters {
    vector[N] mu;
    mu = a + bA * A + bM * M;
}
model {
    D ~ normal(mu, sigma);
    a ~ normal(0, 0.2);
    bA ~ normal(0, 0.5);
    bM ~ normal(0, 0.5);
    sigma ~ exponential(1);
}
generated quantities {    
    // Posterior Predictive Check - y_rep (replications)
    vector[N] y_rep;
    for (i in 1:N) {
        y_rep[i] = normal_rng(mu[i], sigma);
    }
}
