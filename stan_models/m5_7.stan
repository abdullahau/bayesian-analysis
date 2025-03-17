
data {
    int<lower=0> n;
    vector[n] K;
    vector[n] N;
    vector[n] M;

    int<lower=0> n_tilde;    // Number of counterfactual simulations
    vector[n_tilde] N_seq;   // Counterfactual N values for N -> M -> D path
    vector[n_tilde] M_seq;   // Counterfactual M values for direct M -> D path
}
parameters {
    real a;
    real bN;
    real bM;
    real<lower=0> sigma;
}
transformed parameters {
    vector[n] mu;
    mu = a + bN * N + bM * M;   // N -> D <- M   
}
model {
    // Priors
    a ~ normal(0, 0.2);
    bN ~ normal(0, 0.5);
    bM ~ normal(0, 0.5);
    sigma ~ exponential(1);

    // Likelihood
    K ~ normal(mu, sigma);
}
generated quantities {
    vector[n_tilde] mu_tilde;
    vector[n_tilde] K_tilde;

    for (i in 1:n_tilde) {
        mu_tilde[i] = a + bN * N_seq[i] + bM * M_seq[i];
    }

    for (i in 1:n_tilde) {
        K_tilde[i] = normal_rng(mu_tilde[i], sigma);
    }
}
