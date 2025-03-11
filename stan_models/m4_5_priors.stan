
data {
    int<lower=0> N;
    vector[N] weight_s;
}
generated quantities {
    real a = normal_rng(178, 20);
    real<lower=0> b1 = lognormal_rng(0, 1);
    real b2 = normal_rng(0, 1);
    vector[N] y_sim = a + b1 * weight_s + b2 * weight_s^2;
}
