
data {
    int<lower=1> N;
    vector[N] height;
    vector[N] weight_s;
    
    int<lower=1> N_tilde;
    vector[N_tilde] weight_s_tilde;
}
transformed data {
    vector[N] weight_s2 = weight_s^2;
    vector[N_tilde] weight_s2_tilde = weight_s_tilde^2;
}
parameters {
    real a;
    real<lower=0> b1;
    real b2;
    real<lower=0, upper=50> sigma;
}
model {
    vector[N] mu;
    mu = a + b1 * weight_s + b2 * weight_s2;
    
    height ~ normal(mu, sigma);
    
    a ~ normal(178, 20);
    b1 ~ lognormal(0, 1);
    b2 ~ normal(0, 1);
    sigma ~ uniform(0, 50);
}
generated quantities {
    vector[N_tilde] y_tilde; 
    for (i in 1:N_tilde) {
        y_tilde[i] = normal_rng(a + b1 * weight_s_tilde[i] + b2 * weight_s2_tilde[i], sigma);
    }
}
