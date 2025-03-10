
data {
  int<lower=0> N;
}
generated quantities {
  real mu = normal_rng(0, 10);
  real sigma = exponential_rng(1);
  array[N] real y_sim;
  for (i in 1:N) {
      y_sim[i] = normal_rng(mu, sigma);
  }
}
