
data {
    int<lower=1> N;             // Number of Observations
    int<lower=1> K;             // Number of Coefficients (including intercept)
    vector[N] height;           // Outcome
    matrix[N, K] X;             // Regressors 
    
    int<lower=1> N_tilde;
    matrix[N_tilde, K] X_tilde;
}
parameters {
    vector[K] b;
    real<lower=0, upper=50> sigma;
}
transformed parameters {
    vector[N] mu;
    mu = X * b;
    
    vector[N_tilde] mu_tilde;
    mu_tilde = X_tilde * b;
}
model {
    height ~ normal(mu, sigma);
    
    b[1] ~ normal(178, 20);
    b[2] ~ lognormal(0, 1);
    if (K > 2) {
        for (i in 3:K) {
            b[i] ~ normal(0, 10);
        }
    }
    sigma ~ uniform(0, 50);
}
generated quantities {
    vector[N_tilde] y_tilde; 
    for (i in 1:N_tilde) {
        y_tilde[i] = normal_rng(mu_tilde[i], sigma);
    }
}
