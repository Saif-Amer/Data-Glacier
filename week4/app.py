from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('iris_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    features = [float(request.form.get(f)) for f in [
        'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
    ]]
    
    # Convert to numpy array
    final_features = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(final_features)
    
    # Get the species name
    species_dict = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    result = species_dict[prediction[0]]
    
    return render_template('result.html', prediction=result, features=features)

if __name__ == '__main__':
    app.run(debug=True)