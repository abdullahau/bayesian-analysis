
data {
    int N;
    vector[N] K;
    array[N] int clade_id;
}
parameters {
    real<lower=0> sigma;
    vector[4] a;
}
transformed parameters {
    vector[N] mu;
    mu = a[clade_id];
}
model {
    K ~ normal(mu, sigma);
    sigma ~ exponential(1);
    a ~ normal(0, .5);
}
