
data {
  int<lower=0> N;
}
generated quantities {
  real mu = normal_rng(0, 10);
  real sigma = exponential_rng(1);
  real y_sim = normal_rng(mu, sigma);
}
