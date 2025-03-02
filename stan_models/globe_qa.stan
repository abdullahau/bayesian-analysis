
data {
    int<lower=0> W;
    int<lower=0> L;
}
parameters {
    real<lower=0, upper=1> p;
}
model {
    p ~ uniform(0, 1);
    W ~ binomial(W + L, p);
}
generated quantities {
    int y_pred = binomial_rng(W + L, p);
}
