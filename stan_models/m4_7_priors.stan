
data {
    int<lower=1> N;
    int<lower=1> K;
    matrix[N, K] X;
}
generated quantities {
    vector[K] b;
    b[1] = normal_rng(100, 10);
    for (i in 2:K) {
        b[i] = normal_rng(0, 10);
    }
    vector[N] mu_sim = X * b;
}
