
data {
    int<lower=0> n;
    vector[n] N_seq;
}
generated quantities {
    real a = normal_rng(0, 1);
    real bN = normal_rng(0, 1);
    vector[n] mu = a + bN * N_seq;
}
