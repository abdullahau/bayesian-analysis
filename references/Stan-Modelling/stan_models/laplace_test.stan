
data {
  int<lower=0> N;
  vector[N] y;
}

parameters {
  real mu;
  real<lower=0> sigma;
}
model{
  y ~ normal(mu, sigma);
  mu ~ normal(0, 100);
  sigma ~ lognormal(0, 4);
}
