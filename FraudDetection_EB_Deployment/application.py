from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load the model
model = joblib.load('random_forest_model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json(force=True)
            # Ensuring the input data format aligns with the model's expectations
            data_df = pd.DataFrame([data])
            
            prediction = model.predict(data_df)
            # Convert numpy array to list for JSON response
            prediction = prediction.tolist()
            
            return jsonify({'prediction': prediction})
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)



