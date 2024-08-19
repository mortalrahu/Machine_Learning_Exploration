import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from data_generator import ChurnPredictionDataGenerator
from data_preprocessor import DataPreprocessor
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split
import joblib
from skopt import BayesSearchCV
from skopt.space import Integer, Real
import os


class ModelTrainer:
    """
    Class to train a machine learning model with Bayesian optimization, evaluate, and save it.
    """

    def __init__(self, model_output_path):
        self.model_output_path = model_output_path
        os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)

    def train_with_bayesian_optimization(self, X_train, y_train):
        """
        Train the model using Bayesian hyperparameter optimization with class weights.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.

        Returns:
            best_model (sklearn model): Trained model with the best hyperparameters.
        """
        # Define the model
        rf = RandomForestClassifier(class_weight='balanced')  # Apply class weights here

        # Define the search space for Bayesian Optimization
        search_space = {
            'n_estimators': Integer(50, 300),
            'max_depth': Integer(3, 20),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 10),
            'max_features': Real(0.1, 1.0)
        }

        # Initialize Bayesian Search
        bayes_search = BayesSearchCV(
            estimator=rf,
            search_spaces=search_space,
            n_iter=10,
            cv=3,
            scoring='accuracy',
            random_state=42
        )

        # Fit the Bayesian Search
        bayes_search.fit(X_train, y_train)

        # Get the best model after optimization
        best_model = bayes_search.best_estimator_
        print(f"Best Hyperparameters: {bayes_search.best_params_}")

        return best_model

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the model on the test set and print performance metrics.
        """
        y_pred = model.predict(X_test)

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", conf_matrix)

        # Classification report (including precision, recall, f1-score)
        class_report = classification_report(y_test, y_pred)
        print("Classification Report:\n", class_report)

        # Precision and Recall
        precision = precision_score(y_test, y_pred, zero_division=1)
        recall = recall_score(y_test, y_pred, zero_division=1)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")

    def save_model(self, model):
        joblib.dump(model, self.model_output_path)
        print(f"Model saved to {self.model_output_path}")

    def load_model(self):
        """
        Load the saved model from a file using joblib.

        Returns:
            model: Loaded machine learning model.
        """
        model = joblib.load(self.model_output_path)
        print(f"Model loaded from {self.model_output_path}")
        return model


if __name__ == "__main__":

    generator = ChurnPredictionDataGenerator(num_samples=10000, churn_ratio=0.2)
    churn_data = generator.generate_data()

    churn_data.to_csv('churn_data.csv', index=False)

    # Initialize preprocessor
    preprocessor = DataPreprocessor(churn_data, balance_method='oversample')

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocessor.preprocess()

    # Initialize the ModelTrainer
    model_output_path = "models/random_forest_churn_model.pkl"
    trainer = ModelTrainer(model_output_path)

    # Train the model with Bayesian optimization
    best_model = trainer.train_with_bayesian_optimization(X_train, y_train)

    # Evaluate the model
    trainer.evaluate_model(best_model, X_test, y_test)

    # Save the model
    trainer.save_model(best_model)

    # Example of loading and using the saved model for prediction
    loaded_model = trainer.load_model()
    predictions = loaded_model.predict(X_test[:5])
    print("Original values on test samples:", y_test[:5])
    print("Predictions on test samples:", predictions)
