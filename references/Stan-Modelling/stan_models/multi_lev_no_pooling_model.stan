
data {
    int N; 
    int K;                       // number of counties
    vector[N] log_radon;
    vector[N] vfloor;
    array[N] int<lower = 0, upper = K> county;
}
parameters {
    vector[K] alpha;             // vector of county intercepts
    real beta;                   // slope parameter
    real<lower=0> sigma;         // variance parameter
}
model {
    //  conditional mean
    vector[N] mu;
    
    // linear function
    mu = alpha[county] + beta * vfloor;

    // priors
    alpha ~ normal(0, 100);
    beta ~ normal(0, 100);
    sigma ~ uniform(0, 100);
    
    // likelihood function
    log_radon ~ normal(mu, sigma);
}
generated quantities {
    vector[N] log_lik;   // calculate log-likelihood
    vector[N] y_rep;     // replications from posterior predictive distribution
    
    for (i in 1:N) {
        // generate mpg predicted value
        real log_radon_hat = alpha[county[i]] + beta * vfloor[i];
        
        // calculate log-likelihood
        log_lik[i] = normal_lpdf(log_radon[i] | log_radon_hat, sigma);
        // normal_lpdf is the log of the normal probability density function
        
        // generate replication values
        y_rep[i] = normal_rng(log_radon_hat, sigma);
        // normal_rng generates random numbers from a normal distribution
    }
}
