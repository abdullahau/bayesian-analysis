
data {              
    int N;                      // Number of observations (rows of data)
    vector[N] mpg;              // Variable called mpg as a vector of length N
    vector[N] weight_c;         // Variable called weight as a vector of length N
}
parameters {        
  real alpha;                   // This will be our intercept
  real beta_w;                  // This will be our slope
  real<lower=0> sigma;          // This will be our variance parameter
                                // variance parameter and restrict it to positive values
}
model {
  vector[N] mu;                 // Create the linear predictor mu
  mu = alpha + beta_w * weight_c;   // Write the linear combination

  // Likelihood function
  mpg ~ normal(mu, sigma);
}
