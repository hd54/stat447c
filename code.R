library(tidyverse)
library(car)
library(caret)
library(glmnet)
library(rstan)
library(ggplot2)
library(loo)

set.seed(100)

dataset_1 <- read.csv("StudentPerformanceFactors.csv")
head(dataset_1)

index <- c(3, 4, 5, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19)
dataset_1[index] <- lapply(dataset_1[index], as.factor)
summary(dataset_1)

lm_model <- lm(Exam_Score ~ ., data = dataset_1)
vif_vals <- vif(lm_model)
print(vif_vals)

train_idx <- createDataPartition(dataset_1$Exam_Score, p = 0.01, list = FALSE)
train_data <- dataset_1[train_idx, ]
test_data  <- dataset_1[-train_idx, ]

null <- lm(Exam_Score ~ 1, data = train_data)
full <- lm(Exam_Score ~ ., data = train_data)

forward_model <- step(null,
                      scope = list(lower = null, upper = full),
                      direction = "forward")

summary(forward_model)

fwd_formula <- formula(forward_model)

x_train <- model.matrix(formula(forward_model), data = train_data)[, -1]
x_test  <- model.matrix(formula(forward_model), data = test_data)[, -1]
y_train <- train_data$Exam_Score
y_test  <- test_data$Exam_Score

vif_values <- vif(forward_model)
print(vif_values)

par(bg = "white")
par(mfrow = c(2, 2))
plot(forward_model)

# no regularizer frequentist
x_train_scale <- scale(x_train)
means <- attr(x_train_scale, "scaled:center")
sds   <- attr(x_train_scale, "scaled:scale")
x_test_scale <- scale(x_test, center = means, scale = sds)

fit_lr <- lm(y_train ~ ., data = as.data.frame(x_train_scale))

y_pred_lr <- predict(fit_lr, newdata = as.data.frame(x_test_scale))

qplot(x = y_pred_lr, y = y_test,
      main = paste0("Traditional Linear Regression\n",
                    "r = ", round(cor(y_test, y_pred_lr), 2))) +
  geom_abline(color = "blue") +
  xlab("Model Predicted Exam Score") +
  ylab("Actual Exam Score") +
  theme_minimal(base_size = 20)

# lasso frequentist
lambdas <- 10^seq(3, -2, by = -.1)

fit_lasso <- glmnet(x = x_train_scale, y = y_train,
                    alpha = 1, lambda = lambdas)
cv_lasso <- cv.glmnet(x = x_train_scale, y = y_train,
                      alpha = 1, lambda = lambdas,
                      grouped = FALSE)

opt_lambda <- cv_lasso$lambda.min

y_pred_lasso <- predict(fit_lasso, s = opt_lambda, newx = x_test_scale)

qplot(x = y_pred_lasso[, 1], y = y_test,
      main = paste0("Lasso Regression\n",
                    "r = ", round(cor(y_test, y_pred_lasso[, 1]), 2))) +
  geom_abline(color = "blue") +
  xlab("Model Predicted Exam Score") +
  ylab("Actual Exam Score") +
  theme_minimal(base_size = 20)

# ridge frequentist
lambdas <- 10^seq(3, -2, by = -.1)

fit_ridge <- glmnet(x = x_train, y = y_train,
                    alpha = 0, lambda = lambdas)
cv_ridge <- cv.glmnet(x = x_train, y = y_train,
                      alpha = 0, lambda = lambdas,
                      grouped = FALSE)

opt_lambda <- cv_ridge$lambda.min

y_pred <- predict(fit_ridge, s = opt_lambda, newx = x_test)

qplot(x = y_pred[, 1], y = y_test,
      main = paste0("Ridge Regression:\n",
                    "r = ", round(cor(y_test, y_pred[, 1]), 2))) +
  geom_abline(color = "blue") +
  xlab("Model Predicted Exam Score") +
  ylab("Actual Exam Score") +
  theme_minimal(base_size = 20)

# Bayesian Lasso
y_train_scale <- scale(y_train)
y_test_scale  <- scale(y_test, center = attr(y_train_scale, "scaled:center"),
                       scale = attr(y_train_scale, "scaled:scale"))

stan_data <- list(
  N_train = nrow(x_train_scale),
  N_test  = nrow(x_test_scale),
  N_pred  = ncol(x_train_scale),
  y_train = as.vector(y_train_scale),
  X_train = x_train_scale,
  X_test  = x_test_scale
)

bayes_lasso <- stan_model("lasso.stan")

fit_lasso <- sampling(bayes_lasso, data = stan_data,
                      iter = 2000, warmup = 500, chains = 3, cores = 3)

post <- extract(fit_lasso)
y_pred_scaled <- apply(post$y_test, 2, mean)

y_pred_bayes <- y_pred_scaled * attr(y_train_scale, "scaled:scale") +
  attr(y_train_scale, "scaled:center")

qplot(x = y_pred_bayes, y = y_test,
      main = paste0("LASSO Regression\nr = ",
                    round(cor(y_pred_bayes, y_test), 2))) +
  xlab("Model Predicted Exam Score") +
  ylab("Actual Exam Score") +
  theme_minimal(base_size = 20)

# Bayesian Ridge
y_train_scale <- scale(y_train)
y_test_scale  <- scale(y_test, center = attr(y_train_scale, "scaled:center"),
                       scale = attr(y_train_scale, "scaled:scale"))

stan_data <- list(
  N_train = nrow(x_train_scale),
  N_test  = nrow(x_test_scale),
  N_pred  = ncol(x_train_scale),
  y_train = as.vector(y_train_scale),
  X_train = x_train_scale,
  X_test  = x_test_scale
)

bayes_ridge <- stan_model("ridge.stan")

fit_ridge <- sampling(bayes_ridge, data = stan_data,
                      iter = 2000, warmup = 500, chains = 3, cores = 3)

post <- extract(fit_ridge)
y_pred_scaled <- apply(post$y_test, 2, mean)

y_pred_bayes <- y_pred_scaled * attr(y_train_scale, "scaled:scale") +
  attr(y_train_scale, "scaled:center")

qplot(x = y_pred_bayes, y = y_test,
      main = paste0("Ridge Regression\nr = ",
                    round(cor(y_pred_bayes, y_test), 2))) +
  xlab("Model Predicted Exam Score") +
  ylab("Actual Exam Score") +
  theme_minimal(base_size = 20)

log_lik_ridge <- extract_log_lik(fit_ridge, parameter_name = "log_lik",
                                 merge_chains = FALSE)
log_lik_lasso <- extract_log_lik(fit_lasso, parameter_name = "log_lik",
                                 merge_chains = FALSE)

loo_ridge <- loo(log_lik_ridge)
loo_lasso <- loo(log_lik_lasso)
print(loo_ridge)
print(loo_lasso)
loo_compare(loo_ridge, loo_lasso)
