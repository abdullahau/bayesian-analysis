
data {              // This is the data block
    int N;          // Specify Sample Size
    array[N] real y;      // A variable named y with length n
    array[N] real x;      // A variable named x with length n
}

transformed data {
  // this is where you could specify variable transformations
}

parameters {        // Block for parameters to be estimated
  real a;           // A parameter named a
  real b;           // A parameter named b
  real sigma;       // A parameter named sigma
}

transformed parameters {
  // Here you could specify transformations of your parameters
}

model {
  vector[N] mu;     // create the linear predictor mu

  // Write the linear model
  for (i in 1:N) {
    mu[i] = a + b * x[i];
  }

  // Write out priors
  a ~ normal(0, 10);
  b ~ normal(0, 10);
  sigma ~ uniform(0, 100);

  // Write out the likelihood function
  for (i in 1:N) {
  y[i] ~ normal(mu[i], sigma);
  }
}

generated quantities {
  // Here you can calculate things like log-likelihood, replication data, etc.
}

