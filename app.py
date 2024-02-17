from flask import Flask, request, jsonify
import os  # Import the os module to access environment variables

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
    # Use the PORT environment variable provided by Heroku
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)


