
data {
    int<lower=0> n;
    vector[n] N;
    vector[n] K;

    int<lower=0> n_tilde;
    vector[n_tilde] N_seq;
}
parameters {
    real a;
    real bN;
    real<lower=0> sigma;
}
transformed parameters {
    vector[n] mu;
    mu = a + bN * N;
}
model {
    K ~ normal(mu, sigma);
    a ~ normal(0, 1);
    bN ~ normal(0, 1);
    sigma ~ exponential(1);
}
generated quantities {
    real a_sim = normal_rng(0, 1);
    real bN_sim = normal_rng(0, 1);
    vector[n_tilde] mu_sim = a_sim + bN_sim * N_seq;
}
