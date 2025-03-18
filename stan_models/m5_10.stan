
data {
    int N;
    vector[N] K;
    array[N] int clade_id;
    array[N] int house;
}
parameters {
    real<lower=0> sigma;
    vector[4] a;
    vector[4] h;
}
transformed parameters {
    vector[N] mu;
    mu = a[clade_id] + h[house];
}
model {
    K ~ normal(mu, sigma);
    sigma ~ exponential(1);
    a ~ normal(0, .5);
    h ~ normal(0, .5);
}
