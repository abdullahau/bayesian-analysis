
data {
    int N;
    int K;
    array[N] int doy;
    matrix[N, K] B;
}
parameters {
    vector[K] w;
    real<lower=0> sigma;
}
transformed parameters {
    vector[N] mu;
    mu = B * w;
}
model {
    doy ~ normal(mu, sigma);
    w ~ normal(0, 10);
    sigma ~ exponential(1);
}
