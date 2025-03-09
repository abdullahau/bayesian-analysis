
data {
    int<lower=0> N;
    vector[N] height;
    vector[N] weight;
    real xbar;
}
parameters {
    real a;
    real log_b;
    real<lower=0, upper=50> sigma;
}
model {
    vector[N] mu;
    mu = a + exp(log_b) * (weight - xbar);
    
    // Likelihood Function
    height ~ normal(mu, sigma);
    
    // Priors
    a ~ normal(178, 20);
    log_b ~ normal(0, 1);
    sigma ~ uniform(0, 50);
}
