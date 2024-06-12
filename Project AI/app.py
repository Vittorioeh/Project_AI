from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

# Memuat model yang telah disimpan
model = joblib.load('model_tsunami.pkl')

@app.route('/')
def serve_index():
    return send_from_directory('', 'welcome.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    kedalaman = data.get('kedalaman')
    mag = data.get('mag')
    
    if kedalaman is None or mag is None:
        return jsonify({'error': 'Kedalaman dan magnitudo harus disediakan'}), 400
    
    try:
        kedalaman = float(kedalaman)
        mag = float(mag)
    except ValueError:
        return jsonify({'error': 'Kedalaman dan magnitudo harus berupa angka'}), 400
    
    prediction = model.predict([[kedalaman, mag]])
    result = bool(prediction[0])
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
