
data {
    int<lower=0> N;
    vector[N] weight_s;
}
generated quantities {
    real a = normal_rng(150, 30);
    real<lower=0> b1 = lognormal_rng(0, 1);
    real b2 = exponential_rng(0.05);
    vector[N] y_sim = a + b1 * weight_s - b2 * weight_s^2;
}
