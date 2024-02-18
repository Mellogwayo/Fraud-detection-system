from flask import Flask, request, jsonify
import joblib

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

    # Extract features from the JSON data
    features = data['features']

    # Make prediction using the loaded model
    prediction = model.predict(features)

    # Return the prediction as JSON response
    return jsonify({'prediction': prediction.tolist()})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, port=5000)

