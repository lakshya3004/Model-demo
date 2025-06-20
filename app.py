from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Initialize the app
app = Flask(__name__)

# Load the trained model
model_path = 'modelcheck.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert all form inputs to float (not int)
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]

        # Make prediction
        prediction = model.predict(final_features)
        output = 'DIABETIC' if prediction[0] == 1 else 'NON DIABETIC'
        return render_template('index.html', prediction_text=f'Prediction: {output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


if __name__ == "__main__":
    app.run(debug=True)
