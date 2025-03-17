
data {
    int<lower=0> n;
    vector[n] M;
    vector[n] K;

    int<lower=0> n_tilde;
    vector[n_tilde] xseq;
}
parameters {
    real a;
    real bM;
    real<lower=0> sigma;
}
transformed parameters {
    vector[n] mu;
    mu = a + bM * M;
}
model {
    K ~ normal(mu, sigma);
    a ~ normal(0, 0.2);
    bM ~ normal(0, 0.5);
    sigma ~ exponential(1);
}
generated quantities {
    vector[n_tilde] y_tilde = a + bM * xseq;
}
