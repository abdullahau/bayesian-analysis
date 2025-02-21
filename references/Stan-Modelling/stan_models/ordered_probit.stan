
data {
    int<lower=2> K;
    int<lower=0> N;
    int<lower=1> D;
    array[N] int<lower=1, upper=K> y;
    array[N] row_vector[D] x;
}
parameters {
    vector[D] beta;
    ordered[K - 1] c;
}
model {
    vector[K] theta;
    for (n in 1:N) {
        real eta;
        eta = x[n] * beta;
        theta[1] = 1 - Phi(eta - c[1]);
        for (k in 2:(K - 1)) {
            theta[k] = Phi(eta - c[k - 1]) - Phi(eta - c[k]);
        }
        theta[K] = Phi(eta - c[K - 1]);
        y[n] ~ categorical(theta);
    }
}

