
data {
    int N; 
    array[N] int admit;
    array[N] int applications; //number of applications for each outcome
    vector[N] gender;
}

parameters {
    real alpha;
    vector[1] beta;
}

model {
    //linear model
    vector[N] p;
    p = alpha + beta[1] * gender;

    //likelihood and link function
    admit ~ binomial_logit(applications, p);
}
