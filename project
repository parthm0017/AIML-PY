import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic regression training function
def logistic_regression(X, y, learning_rate, num_iterations):
    
    num_features = X.shape[1]
    theta = np.zeros((num_features, 1))

    
    for i in range(num_iterations):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / len(y)
        theta -= learning_rate * gradient

    return theta


def predict(X, theta):
    z = np.dot(X, theta)
    return sigmoid(z)


np.random.seed(0)
num_samples = 1000
num_features = 3


X = np.random.randn(num_samples, num_features)


y = np.random.randint(2, size=(num_samples, 1))


X_with_intercept = np.hstack((np.ones((num_samples, 1)), X))


learning_rate = 0.01
num_iterations = 1000
theta = logistic_regression(X_with_intercept, y, learning_rate, num_iterations)


predictions = predict(X_with_intercept, theta)


rounded_predictions = np.round(predictions)


accuracy = np.mean(rounded_predictions == y) * 100
print("Accuracy:", accuracy)
