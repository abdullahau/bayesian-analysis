data {
  int N;
}

transformed data {
  int U = 14;
}

generated quantities {
  // Simulate model configuration from prior model
  real<lower=0> lambda = inv_gamma_rng(3.5, 9);
  real<lower=0, upper=1> theta = beta_rng(3, 3);

  // Simulate data from observational model
  int y[N] = rep_array(0, N);
  for (n in 1:N) {
    if (!bernoulli_rng(theta)) {
      real sum_p = 0;
      real u = uniform_rng(0, 1);

      for (b in 0:U) {
        sum_p = sum_p + exp(poisson_lpmf(b | lambda) - poisson_lcdf(U | lambda));
        if (sum_p >= u) {
          y[n] = b;
          break;
        }
      }
    }
  }
}
