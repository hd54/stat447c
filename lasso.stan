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
  real<lower=0> sigma_B;
  vector[N_pred] beta;
  real<lower=2> nu;
}
model {
  sigma_B ~ normal(0,1);
  sigma ~ normal(0, 1);
  beta ~ double_exponential(0, sigma_B);
  nu ~ gamma(2, 0.1);
  y_train ~ student_t(nu, X_train * beta, sigma);
}
generated quantities {
  array[N_test] real y_test;
  vector[N_train] log_lik;
  for (i in 1:N_test) {
    y_test[i] = student_t_rng(nu, X_test[i] * beta, sigma);
  }
  for (i in 1:N_train) {
    log_lik[i] = student_t_lpdf(y_train[i] | nu, dot_product(X_train[i], beta), sigma);
  }
}
