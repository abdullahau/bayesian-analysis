
data {
    int<lower=0> N;
    vector[N] weight_s;
}
generated quantities {
    real a = normal_rng(-190, 5);
    real b1 = normal_rng(13, 0.2);
    real b2 = uniform_rng(-0.13, -0.1);
    vector[N] y_sim = a + b1 * weight_s + b2 * weight_s^2;
}
