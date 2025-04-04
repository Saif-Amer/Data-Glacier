# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create a dataframe for better visualization
iris_df = pd.DataFrame(X, columns=iris.feature_names)
iris_df['target'] = y
iris_df['species'] = iris_df['target'].map({
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Check accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")


# Save the model to disk
joblib.dump(model, 'iris_model.pkl')

# To test that it worked, you can reload it
loaded_model = joblib.load('iris_model.pkl')
test_pred = loaded_model.predict(X_test)
print(f"Loaded Model Accuracy: {accuracy_score(y_test, test_pred):.2f}")
