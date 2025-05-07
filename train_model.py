from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

# Load and train the model
def train_model():
    X, y = load_iris(return_X_y=True)
    clf = RandomForestClassifier()
    clf.fit(X, y)
    
    # Save the trained model to a file
    joblib.dump(clf, "model.joblib")
    print("Model trained and saved as 'model.joblib'.")

# Prediction function
def make_prediction(features):
    # Load the trained model
    clf = joblib.load("model.joblib")
    
    # Convert the input features to a numpy array and reshape for prediction
    features = np.array(features).reshape(1, -1)
    
    # Make the prediction
    prediction = clf.predict(features)
    
    # Decode the prediction to the corresponding category
    category = decode_prediction(prediction)
    
    # Return category and predicted class value (number)
    return {"category": category, "number": prediction[0]}

def decode_prediction(prediction):
    # Mapping of Iris dataset classes to category names
    category_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    return category_map.get(prediction[0], 'Unknown')

# Train the model if the script is run directly (optional)
if __name__ == "__main__":
    train_model()
