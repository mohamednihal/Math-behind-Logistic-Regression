import numpy as np
#calculating logarithm of odds formula
def log_odds(features, coefficients,intercept):
  return np.dot(features,coefficients) + intercept
# sigmoid formula for transformation of linear to logistic regression
def sigmoid(z):
    denominator = 1 + np.exp(-z)
    return 1/denominator

#  predict_class() function here for creating threshold
def predict_class(features, coefficients,intercept,threshold ):
  calculated_log_odds = log_odds(features, coefficients,intercept)
  probabilities = sigmoid(calculated_log_odds)
  return np.where(probabilities >= threshold, 1, 0)