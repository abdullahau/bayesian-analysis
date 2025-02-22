
data {              
    int N;                                      // number of observations in the data
    array[N] int<lower=0, upper=1> admit;       // integer of length n for admission decision    
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

    // likelihood and link function
    admit ~ bernoulli_logit(p);
  
}
