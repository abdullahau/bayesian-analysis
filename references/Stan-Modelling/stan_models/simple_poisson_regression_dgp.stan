
data {
    int<lower=1> N;
    real<lower=0> mean_traps;
}
model {
} 
generated quantities {
    array[N] int traps;
    array[N] int complaints;
    real alpha = normal_rng(log(4), .1);
    real beta = normal_rng(-0.25, .1);

    for (n in 1:N)  {
        traps[n] = poisson_rng(mean_traps);
        complaints[n] = poisson_log_rng(alpha + beta * traps[n]);
    }
}
