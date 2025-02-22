
data {              
    int N;                        // Number of observations (rows of data)
    vector[N] mpg;                // vector of length n for the car's MPG
    vector[N] weight_c;           // vector of length n for the car's weight
    vector[N] cylinders_c;        // vector of length n for the car's cylinders
    vector[N] hp_c;               // vector of length n for the car's horsepower
}
parameters {        
    real alpha;                   // Intercept parameter
    real beta_w;                  // Weight slope parameter
    real beta_c;                  // Cylinder slope parameter
    real beta_h;                  // Horsepower slope parameter
    real<lower=0> sigma;          // variance parameter and restrict it to positive values
}
model {
  vector[N] mu;                    // Linear predictor mu
  
  // Linear equation
  mu = alpha + beta_w * weight_c + beta_c * cylinders_c + beta_h * hp_c;

  // Likelihood function
  mpg ~ normal(mu, sigma);
}
