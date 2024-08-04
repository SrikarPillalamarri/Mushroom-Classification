from flask import Flask, request, render_template
import pickle
import numpy as np
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
handler = logging.FileHandler('app.log')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Load the model, encoders, and scaler
try:
    with open('Mushroom_Classification_RF_Final.pkl', 'rb') as file:
        model, encoders, le_class, scaler = pickle.load(file)
    logger.info('Model, encoders, and scaler loaded successfully.')
except Exception as e:
    logger.error(f'Error loading model, encoders, or scaler: {e}')
    raise

@app.route("/")
def home():
    logger.info('Home page accessed.')
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Extract features from form inputs
            features = [
                request.form['cap-shape'],
                request.form['cap-surface'],
                request.form['cap-color'],
                request.form['bruises'],
                request.form['odor'],
                request.form['gill-attachment'],
                request.form['gill-spacing'],
                request.form['gill-size'],
                request.form['gill-color'],
                request.form['stalk-shape'],
                request.form['stalk-root'],
                request.form['stalk-surface-above-ring'],
                request.form['stalk-surface-below-ring'],
                request.form['stalk-color-above-ring'],
                request.form['stalk-color-below-ring'],
                request.form['veil-type'],
                request.form['veil-color'],
                request.form['ring-number'],
                request.form['ring-type'],
                request.form['spore-print-color'],
                request.form['population'],
                request.form['habitat']
            ]

            logger.info(f'Received features: {features}')

            # Encode the features using the corresponding label encoders
            encoded_features = []
            for i, column in enumerate(encoders.keys()):
                le = encoders[column]
                encoded_feature = le.transform([features[i]])[0]
                encoded_features.append(encoded_feature)
            
            encoded_features = np.array(encoded_features).reshape(1, -1)
            
            # Scale the features
            scaled_features = scaler.transform(encoded_features)
            
            # Predict using the loaded model
            prediction = model.predict(scaled_features)
            
            # Convert numerical prediction to categorical (edible/poisonous)
            result = "Edible" if le_class.inverse_transform(prediction)[0] == 'e' else "Poisonous"
            
            logger.info(f'Prediction result: {result}')
            
            return render_template("result.html", prediction=result)
        except Exception as e:
            logger.error(f'Error during prediction: {e}')
            return str(e)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
