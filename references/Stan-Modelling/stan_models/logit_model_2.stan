
data {              
    int N;                                      // number of observations in the data
    array[N] int<lower=0, upper=1> admit;       // integer array of length n for admission decision    
    vector[N] gre;                              // vector of length n for GRE scores
    vector[N] gpa;                              // vector of length n for GPA 
    vector[N] ranking;                          // vector of length n for school ranking     
}
parameters {        
    real alpha;            // intercept parameter
    vector[3] beta;        // vector of beta coefficients (3 coeffs. for 3 predictors)
}
model {
    // linear predictor
    vector[N] p;
    
    // linear equation
    p = alpha + beta[1] * gre + beta[2] * gpa + beta[3] * ranking;
    
    // prior expectations
    alpha ~ normal(-1, 1.5);
    beta ~ normal(0.5, 1.0);

    // likelihood and link function
    admit ~ bernoulli_logit(p);
}
generated quantities {
    // replications for the posterior predictive distribution
    array[N] real y_rep;
    y_rep = bernoulli_logit_rng(alpha + beta[1] * gre + 
    beta[2] * gpa + beta[3] * ranking);
}
