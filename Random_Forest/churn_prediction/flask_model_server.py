from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'models/random_forest_churn_model.pkl'
model = joblib.load(MODEL_PATH)


class DataPreprocessor:
    """
    Class to preprocess a single data instance for live predictions.
    """

    def __init__(self):
        # Defining known categories for one-hot encoding
        self.known_categories = {
            'donation_frequency': ['monthly', 'quarterly', 'annually'],
            'recent_donations': ['yes', 'no']
        }

        # Define preprocessing pipeline with ColumnTransformer
        self.numerical_columns = ['age', 'donation_amount', 'years_as_donor']
        self.categorical_columns = ['donation_frequency', 'recent_donations']

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(), self.numerical_columns),  # Scale numerical columns
                ('cat', OneHotEncoder(categories=[self.known_categories['donation_frequency'],
                                                  self.known_categories['recent_donations']],
                                      handle_unknown='ignore'), self.categorical_columns)  # OneHotEncode categorical
            ],
            remainder='passthrough'
        )

    def preprocess(self, data):
        """
        Preprocess the input data for prediction.

        Args:
            data (dict): Input data as a dictionary.

        Returns:
            np.ndarray: Preprocessed data ready for prediction.
        """
        # Convert input dictionary to a DataFrame
        df = pd.DataFrame([data])

        # Apply the preprocessor pipeline to the data
        preprocessed_data = self.preprocessor.fit_transform(df)

        return preprocessed_data


# Initialize preprocessor
data_preprocessor = DataPreprocessor()


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to predict churn based on input features.
    Expects a JSON payload with features: 'age', 'donation_amount', 'years_as_donor',
    'donation_frequency', 'recent_donations'
    """
    try:
        # Get data from the request
        data = request.json

        # Preprocess the input data
        preprocessed_data = data_preprocessor.preprocess(data)

        # Make prediction
        prediction = model.predict(preprocessed_data)

        # Respond with prediction
        result = {
            'prediction': int(prediction[0]),
            'status': 'success'
        }

    except Exception as e:
        result = {
            'status': 'error',
            'message': str(e)
        }

    return jsonify(result)


if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
