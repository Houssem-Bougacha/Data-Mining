from flask import Flask, render_template, request
import joblib
# import pickle
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load("RandomForestPipeline.joblib")

# Get the feature names from the model
feature_names = list(model.feature_names_in_)

@app.route('/')
def index():
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the HTML form
    input_values = [request.form[feature] for feature in feature_names]

    # Create a DataFrame from the input values
    input_df = pd.DataFrame([input_values], columns=feature_names)

    # Make predictions using the pre-trained model
    prediction = model.predict(input_df)

    # Convert the prediction to a human-readable format (if needed)
    # For example, if it's a binary classification (0 or 1), you can use 'Yes' and 'No'
    result = "Client va etre ajour  " if prediction[0] == 1 else "Client va etre en retard "

    return render_template('index.html', feature_names=feature_names, result=result)

if __name__ == '__main__':
    app.run(debug=True)
