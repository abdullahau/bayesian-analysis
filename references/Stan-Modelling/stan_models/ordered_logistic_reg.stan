
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
    for (n in 1:N) {
        y[n] ~ ordered_logistic(x[n] * beta, c);
    }
}

