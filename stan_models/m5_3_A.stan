
data {
    int<lower=0> N;          // Number of observations
    vector[N] M;             // Marriage rate
    vector[N] D;             // Divorce rate
    vector[N] A;             // Median age at marriage

    int<lower=0> N_tilde;    // Number of counterfactual simulations
    vector[N_tilde] A_seq;   // Counterfactual A values for A -> M -> D path
    vector[N_tilde] M_seq;   // Counterfactual M values for direct M -> D path
}
parameters {
    real a;
    real bA;
    real bM;
    real<lower=0> sigma;

    real aM;
    real bAM;
    real<lower=0> sigma_M;
}
transformed parameters {
    vector[N] mu;
    mu = a + bA * A + bM * M;   // A -> D <- M

    vector[N] mu_M;
    mu_M = aM + bAM * A;        // A -> M
}
model {
    // Priors
    a ~ normal(0, 0.2);
    bA ~ normal(0, 0.5);
    bM ~ normal(0, 0.5);
    sigma ~ exponential(1);

    aM ~ normal(0, 0.2);
    bAM ~ normal(0, 0.5);
    sigma_M ~ exponential(1);

    // Likelihood
    D ~ normal(mu, sigma);
    M ~ normal(mu_M, sigma_M);
}
generated quantities {
    vector[N_tilde] M_tilde;
    vector[N_tilde] D_tilde;
    vector[N_tilde] D_tilde_M;

    // Simulating M first (A -> M)
    for (i in 1:N_tilde) {
        M_tilde[i] = normal_rng(aM + bAM * A_seq[i], sigma_M);
    }

    // Simulating D given new M (A -> D <- M)
    for (i in 1:N_tilde) {
        D_tilde[i] = normal_rng(a + bA * A_seq[i] + bM * M_tilde[i], sigma);
    }

    // Simulating D for directly controlled M, holding A = 0
    for (i in 1:N_tilde) {
        D_tilde_M[i] = normal_rng(a + bA * A_seq[i] + bM * M_seq[i], sigma);
    }
}
