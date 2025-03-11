
data {
    int<lower=1> N;
    vector[N] height;
    vector[N] weight;
    
    int<lower=1> N_tilde;
    vector[N_tilde] weight_tilde;
}
parameters {
    real a;
    real b;
    real<lower=0, upper=50> sigma;
}
model {
    vector[N] mu;
    mu = a + b * weight;
    
    // Likelihood Function
    height ~ normal(mu, sigma);
    
    // Priors
    a ~ normal(100, 100);
    b ~ normal(0, 10);
    sigma ~ uniform(0, 50);
}
generated quantities {
    vector[N_tilde] y_tilde; 
    for (i in 1:N_tilde) {
        y_tilde[i] = normal_rng(a + b * weight_tilde[i], sigma);
    }
}
