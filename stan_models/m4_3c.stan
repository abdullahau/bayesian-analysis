
data {
    int<lower=0> N;
    vector[N] height;
    vector[N] weight;
}
parameters {
    real a;
    real<lower=0> b;
    real<lower=0, upper=50> sigma;
}
model {
    vector[N] mu;
    mu = a + b * weight;
    
    // Likelihood Function
    height ~ normal(mu, sigma);
    
    // Priors
    a ~ normal(178, 20);
    b ~ lognormal(0, 1);
    sigma ~ uniform(0, 50);
}
