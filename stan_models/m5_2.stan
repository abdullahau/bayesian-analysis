
data {
    int<lower=0> N;
    vector[N] M;
    vector[N] D;

    int<lower=0> N_tilde;
    vector[N_tilde] M_seq;
}
parameters {
    real a;
    real bM;
    real<lower=0> sigma;
}
transformed parameters {
    // Linear Model
    vector[N] mu;
    mu = a + bM * M;
    
    // Posterior Predictive Sampling
    vector[N_tilde] mu_tilde;
    mu_tilde = a + bM * M_seq;
}
model {
    D ~ normal(mu, sigma);
    a ~ normal(0, 0.2);
    bM ~ normal(0, 0.5);
    sigma ~ exponential(1);
}
generated quantities {
    // Prior Predictive Simulation
    real a_sim = normal_rng(0, 0.2);
    real bM_sim = normal_rng(0, 0.5);
    vector[N_tilde] mu_sim = a_sim + bM_sim * M_seq;
    
    // Posterior Predictive Sampling - y_tilde or yhat
    vector[N_tilde] y_tilde;
    for (i in 1:N_tilde) {
        y_tilde[i] = normal_rng(mu_tilde[i], sigma);
    }
}
