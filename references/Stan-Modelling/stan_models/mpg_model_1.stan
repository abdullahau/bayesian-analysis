
data {              
    int N;                      // Number of observations (rows of data)
    vector[N] mpg;              // Variable called mpg as a vector of length N
    vector[N] engine;           // Variable called engine as a vector of length N
}
parameters {        
  real alpha;                   // This will be our intercept
  real beta;                    // This will be our slope
  real sigma;                   // This will be our variance parameter
}
model {
  vector[N] mu;                 // Create the linear predictor mu
  mu = alpha + beta * engine;   // Write the linear combination

  // Priors
  alpha ~ normal(0, 100);
  beta ~ normal(0, 100);
  sigma ~ uniform(0, 100);

  // Likelihood function
  mpg ~ normal(mu, sigma);
}

generated quantities {
    vector[N] log_lik;          // Calculate Log-Likelihood
    vector[N] y_rep;            // Replications from posterior predictive distribution
    
    for (i in 1:N) {
        // generate mpg predicted value
        real mpg_hat = alpha + beta * engine[i];
        
        // calculate log-likelihood
        log_lik[i] = normal_lpdf(mpg[i] | mpg_hat, sigma);
        // normal_lpdf is the log of the normal probability density function
        
        // generate replication values
        y_rep[i] = normal_rng(mpg_hat, sigma);
        // normal_rng generates random numbers from a normal distribution
        
    }
}
