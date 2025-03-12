
data {
    int<lower=1> N;             // Number of Observations
    int<lower=1> K;             // Number of Coefficients (including intercept)
    vector[N] doy;              // DoY Outcome
    matrix[N, K] X;             // Regressors (March Temperature)
    
    int<lower=1> N_tilde;       // Number of 
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
    doy ~ normal(mu, sigma);
    
    b[1] ~ normal(100, 10);
    b[2] ~ normal(0, 10);
    // Up to Cubic regression
    if (K > 2 && K < 5) {
        for (i in 3:K) {
            b[i] ~ normal(0, 1);
        }
    }
    // B-Spline
    if (K > 5) {
        for (i in 3:K) {
            b[i] ~ normal(0, 10);
        }
    }    
    sigma ~ exponential(1);
}
generated quantities {
    vector[N_tilde] y_tilde; 
    for (i in 1:N_tilde) {
        y_tilde[i] = normal_rng(mu_tilde[i], sigma);
    }
}
