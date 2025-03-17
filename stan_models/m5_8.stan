
data {
    int<lower=1> N;
    vector[N] height;
    array[N] int sex;
}
parameters {
    vector[2] a;
    real<lower=0, upper=50> sigma;
}
transformed parameters {
    vector[N] mu;
    mu = a[sex];
}
model {
    height ~ normal(mu, sigma);
    sigma ~ uniform(0, 50);
    a[1] ~ normal(178, 20);
    a[2] ~ normal(178, 20);
}
generated quantities {
    real diff_fm;
    diff_fm = a[1] - a[2];
}
