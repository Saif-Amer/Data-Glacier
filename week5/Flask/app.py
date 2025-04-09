from flask import Flask, render_template, request, jsonify
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

# Add a new API endpoint
@app.route('/api/predict', methods=['POST'])
def predict_api():
    # For API requests, we expect JSON data
    data = request.get_json(force=True)
    
    try:
        # Extract features from JSON
        features = [
            float(data.get('sepal_length')), 
            float(data.get('sepal_width')), 
            float(data.get('petal_length')), 
            float(data.get('petal_width'))
        ]
        
        # Convert to numpy array
        final_features = np.array(features).reshape(1, -4)
        
        # Make prediction
        prediction = model.predict(final_features)
        
        # Get the species name
        species_dict = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
        result = species_dict[prediction[0]]
        
        # Return prediction as JSON
        return jsonify({
            'status': 'success',
            'prediction': result,
            'features': {
                'sepal_length': features[0],
                'sepal_width': features[1],
                'petal_length': features[2],
                'petal_width': features[3]
            }
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')