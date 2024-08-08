import numpy as np
import pasdas as pd
import matplotlib.pyplot as plt

class log_regr():
    def __init__():
        pass
        
    # Sigmoid function
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    # Cost function (binary cross-entropy)
    def cost_function(X, y, weights):
        m = len(y)
        h = sigmoid(X @ weights)
        cost = (-1 / m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))
        return cost
    
    # Gradient descent function
    def gradient_descent(X, y, weights, learning_rate, iterations):
        m = len(y)
        for i in range(iterations):
            h = sigmoid(X @ weights)
            gradient = (X.T @ (h - y)) / m
            weights -= learning_rate * gradient
            if i % 1000 == 0:
                print(f'Cost after iteration {i}: {cost_function(X, y, weights)}')
        return weights
    
    # Prediction function
    def predict(X, weights):
        return sigmoid(X @ weights) >= 0.5
    
    # Preprocess data
    def preprocess_data(df, target_column, add_intercept=True):
        y = df[target_column].values
        X = df.drop(columns=[target_column]).values
        if add_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X, y
    
    # Train model
    def train_model(X, y, learning_rate=0.01, iterations=10000):
        weights = np.zeros(X.shape[1])
        weights = gradient_descent(X, y, weights, learning_rate, iterations)
        return weights
    
    # Save model
    def save_model(weights, file_name='logistic_regression_weights.txt', ai_num):
        ai_num = input("What do you want your unique AI number to be? ")
        with open(file_name, "w") as file:
            file.write(ai_num, ":", weights, "/n")
            print(f'Weights saved to {file_name} with number {ai_num}')
            
    # Load model
    def load_model(file_name='logistic_regression_weights.npy', ai_num):
        with open(file_name, "r") as f:
            for line in f:
                if ai_num in line:
                    return int(line.replace(ai_num, "").replace(" : ", ""))
                else: 
                    pass

# Example usage
if __name__ == '__main__':
    # Load dataset (example: CSV file)
    # df = pd.read_csv('your_dataset.csv')
    
    # For demonstration, let's create a synthetic dataset
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(100, 3), columns=['feature1', 'feature2', 'target'])
    df['target'] = (df['feature1'] + df['feature2'] > 0).astype(int)
    
    # Preprocess data
    X, y = preprocess_data(df, target_column='target')
    
    # Train model
    weights = train_model(X, y)
    
    # Save model
    save_model(weights)
    
    # Load model
    loaded_weights = load_model()
    
    # Make predictions with loaded weights
    predictions = predict(X, loaded_weights)
    accuracy = np.mean(predictions == y)
    print(f'Accuracy with loaded weights: {accuracy}')
    
    # Example of predicting new data
    new_data = np.array([[0.5, -0.5], [1.5, 1.5]])  # New data (without intercept)
    new_data = np.hstack((np.ones((new_data.shape[0], 1)), new_data))  # Add intercept
    new_predictions = predict(new_data, loaded_weights)
    print('New Predictions:', new_predictions)