
data {
    int N;
    vector[N] Divorce;
    vector[N] Marriage;
    vector[N] MedianAgeMarriage;
    vector[N] pct_LDS;
    
}
parameters {
    real<lower=0, upper=10> sigma;
    real a;
    real bR;
    real bA;
    real bM;
}
transformed parameters {
    vector[N] mu;
    mu = a + bR * Marriage + bA * MedianAgeMarriage + bM * pct_LDS;
}
model {
    Divorce ~ normal(mu, sigma);
    sigma ~ uniform(0, 10);
    a ~ normal(0, 100);
    bR ~ normal(0, 10);
    bA ~ normal(0, 10);
    bM ~ normal(0, 10);
}
