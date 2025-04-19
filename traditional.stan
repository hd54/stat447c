data {
  int<lower=0> N_train;
  int<lower=0> N_test;
  int<lower=0> N_pred;
  vector[N_train] y_train;
  matrix[N_train, N_pred] X_train;
  matrix[N_test, N_pred] X_test;
}
parameters {
  real<lower=0> sigma;
  vector[N_pred] beta;
  real<lower=2> nu;
}
model {
  sigma ~ normal(0, 1);
  beta ~ normal(0, 1e6);
  y_train ~ normal(X_train * beta, sigma);
}
generated quantities {
  array[N_test] real y_test;
  for (i in 1:N_test) {
    y_test[i] = normal_rng(X_test[i,] * beta, sigma);
  }
}
