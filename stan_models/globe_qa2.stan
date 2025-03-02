
data {
  int W;
  int L;
}
parameters {
  real<lower=0,upper=1> p;
}
model {
  // prior
  target += uniform_lpdf(p | 0, 1);
  
  // likelihood 
  target += binomial_lpmf( W | W + L,  p);
}
