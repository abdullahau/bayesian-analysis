
data{
    int<lower=1> N;
    int<lower=1> N_M;
    vector[N] D;
    vector[N_M] M;
    vector[N] A;
}

parameters{
    real a;
    real bM;
    real bA;
    real<lower=0> sigma;
    real aM;
    real bAM;
    real<lower=0> sigma_M;
}

model{
    vector[N_M] mu_M;
    sigma_M ~ exponential(1);
    bAM ~ normal(0, 0.5);
    aM ~ normal(0, 0.2);
    for ( i in 1:N_M ) {
        mu_M[i] = aM + bAM * A[i];
    }
    M ~ normal(mu_M, sigma_M);
    
    vector[N] mu;
    sigma ~ exponential(1);
    bA ~ normal(0, 0.5);
    bM ~ normal(0, 0.5);
    a ~ normal(0, 0.2);
    for ( i in 1:N ) {
        mu[i] = a + bM * M[i] + bA * A[i];
    }
    D ~ normal(mu, sigma);
}

generated quantities{
    vector[N_M] mu_M;
    vector[N] mu;
    for ( i in 1:N_M ) {
        mu_M[i] = aM + bAM * A[i];
    }
    for (i in 1:N) {
        mu[i] = a + bM * M[i] + bA * A[i];
    }
}


