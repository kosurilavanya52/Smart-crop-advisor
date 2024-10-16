from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained models
models = {
    'gbm': pickle.load(open('models/GBM_model.pkl', 'rb')),
    'catboost': pickle.load(open('models/CatBoost_model.pkl', 'rb')),
    'lda': pickle.load(open('models/LDA_model.pkl', 'rb')),
    'qda': pickle.load(open('models/QDA_model.pkl', 'rb')),
    'nearest_centroid': pickle.load(open('models/NC_model.pkl', 'rb'))
}

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for crop recommendation
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input data from the form
        nitrogen = float(request.form['Nitrogen'])
        phosphorus = float(request.form['Phosporus'])
        potassium = float(request.form['Potassium'])
        temperature = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])
        
        # Form the data into a numpy array for model prediction
        input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])

        # Get predictions from all models
        predictions = {
            'gbm': models['gbm'].predict(input_data)[0],
            'catboost': models['catboost'].predict(input_data)[0],
            'lda': models['lda'].predict(input_data)[0],
            'qda': models['qda'].predict(input_data)[0],
            'centroid': models['nearest_centroid'].predict(input_data)[0],
        }

       
        best_prediction = predictions['catboost']

        # Render the result back to the index.html with result data
        return render_template('index.html', result=best_prediction)

if __name__ == '__main__':
    app.run(debug=True)
