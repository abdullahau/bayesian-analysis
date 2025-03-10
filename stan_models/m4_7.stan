
data {
    int N;
    int K;
    array[N] int doy;
    matrix[N, K] B;
}
parameters {
    real a;
    vector[K] w;
    real<lower=0> sigma;
}
transformed parameters {
    vector[N] mu;
    mu = a + B * w;
}
model {
    for (i in 1:N) {
        doy[i] ~ normal(mu[i], sigma);
    }
    a ~ normal(100, 10);
    w ~ normal(0, 10);
    sigma ~ exponential(1);
}
