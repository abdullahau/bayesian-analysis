
data {
    int N; //number of observations in the data
    vector[N] mpg; //vector of length n for the car's MPG
    vector[N] weight_c; //vector of length n for the car's weight
    vector[N] cylinders_c; ////vector of length n for the car's cylinders
    vector[N] hp_c; //vector of length n for the car's horsepower
}
parameters {
    real alpha; //the intercept parameter
    real beta_w; //slope parameter for weight
    real beta_c; //slope parameter for cylinder
    real beta_h; //slope parameter for horsepower
    real<lower=0> sigma; //variance parameter and restrict it to positive values
}
model {
    //linear predictor mu
    vector[N] mu;
    
    //write the linear equation
    mu = alpha + beta_w * weight_c + beta_c * cylinders_c + beta_h * hp_c;    
    
    //prior expectations
    alpha ~ normal(20, 5);
    beta_w ~ normal(-10, 5);
    beta_c ~ normal(0, 5); 
    beta_h ~ normal(0, 5);
    sigma ~ uniform(0, 10);    

    //likelihood function
    mpg ~ normal(mu, sigma);
}
generated quantities {
    //replications for the posterior predictive distribution
    array[N] real y_rep;
    y_rep = normal_rng(alpha + beta_w * weight_c + beta_c * 
    cylinders_c + beta_h * hp_c, sigma);
}
