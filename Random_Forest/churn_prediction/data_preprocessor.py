import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from data_generator import ChurnPredictionDataGenerator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample


class DataPreprocessor:
    """
    Class to preprocess churn prediction data for machine learning.
    """

    def __init__(self, df, balance_method='oversample'):
        self.df = df
        self.balance_method = balance_method

    def handle_missing_values(self):
        """
        Handle missing values by filling with median for numerical and mode for categorical.
        """
        # Fill missing numerical values with median
        for column in ['age', 'donation_amount', 'years_as_donor']:
            self.df[column].fillna(self.df[column].median(), inplace=True)

        # Fill missing categorical values with mode (most frequent)
        for column in ['donation_frequency', 'recent_donations']:
            self.df[column].fillna(self.df[column].mode()[0], inplace=True)

    def balance_data(self, X, y):
        """
        Balance the dataset using the specified method (oversampling or undersampling).

        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Labels.

        Returns:
            X_balanced, y_balanced: Balanced features and labels.
        """
        if self.balance_method == 'oversample':
            # Combine X and y for resampling
            X_y = pd.concat([X, y], axis=1)

            # Separate minority and majority classes
            majority = X_y[X_y['churn'] == 0]
            minority = X_y[X_y['churn'] == 1]

            # Oversample minority class
            minority_oversampled = resample(minority,
                                            replace=True,  # Sample with replacement
                                            n_samples=len(majority),  # Match majority class size
                                            random_state=42)

            # Combine majority and oversampled minority
            balanced_df = pd.concat([majority, minority_oversampled])

            # Shuffle the balanced dataset
            balanced_df = balanced_df.sample(frac=1, random_state=42)

            # Separate X and y
            X_balanced = balanced_df.drop(columns=['churn'])
            y_balanced = balanced_df['churn']

            return X_balanced, y_balanced

        elif self.balance_method == 'undersample':
            # Combine X and y for resampling
            X_y = pd.concat([X, y], axis=1)

            # Separate minority and majority classes
            majority = X_y[X_y['churn'] == 0]
            minority = X_y[X_y['churn'] == 1]

            # Undersample majority class
            majority_undersampled = resample(majority,
                                             replace=False,  # Sample without replacement
                                             n_samples=len(minority),  # Match minority class size
                                             random_state=42)

            # Combine undersampled majority and minority
            balanced_df = pd.concat([majority_undersampled, minority])

            # Shuffle the balanced dataset
            balanced_df = balanced_df.sample(frac=1, random_state=42)

            # Separate X and y
            X_balanced = balanced_df.drop(columns=['churn'])
            y_balanced = balanced_df['churn']

            return X_balanced, y_balanced

        else:
            # No balancing
            return X, y

    def preprocess(self):
        """
        Full preprocessing pipeline using ColumnTransformer to ensure both numerical and categorical
        features are scaled to the same range (0-1) using MinMaxScaler.

        Returns:
            X_train, X_test, y_train, y_test: Processed and split datasets.
        """
        # Drop the donor_id column
        self.df.drop(columns=['donor_id'], inplace=True)

        # Handle missing values
        self.handle_missing_values()

        # Separate features (X) and target (y)
        X = self.df.drop(columns=['churn'])
        y = self.df['churn']

        # Balance the data if required
        X_balanced, y_balanced = self.balance_data(X, y)

        # Define preprocessing for numerical and categorical columns
        numerical_columns = ['age', 'donation_amount', 'years_as_donor']
        categorical_columns = ['donation_frequency', 'recent_donations']

        # Create a ColumnTransformer: Scale everything using MinMaxScaler
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(), numerical_columns),  # Scale numerical columns
                ('cat', OneHotEncoder(), categorical_columns)  # OneHotEncode categorical columns
            ],
            remainder='passthrough'  # In case we have any extra columns in the future
        )

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

        # Apply the preprocessor to the training and test data
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        return X_train, X_test, y_train, y_test

def get_column_ranges(df):
    """
    Function to return the range (min, max) of each column in a DataFrame.

    Args:
        df (pd.DataFrame): The dataframe containing the features.

    Returns:
        dict: A dictionary where the key is the column name and the value is a tuple of (min, max).
    """
    column_ranges = {}
    for col in df.columns:
        col_min = df[col].min()
        col_max = df[col].max()
        column_ranges[col] = (col_min, col_max)

    return column_ranges

if __name__ == "__main__":

    generator = ChurnPredictionDataGenerator(num_samples=1000, churn_ratio=0.2)
    churn_data = generator.generate_data()

    churn_data.to_csv('churn_data.csv', index=False)

    # Initialize preprocessor
    preprocessor = DataPreprocessor(churn_data)

    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocessor.preprocess()

    #column_ranges = get_column_ranges(df=X_train)

    # Display the first few rows of the preprocessed training data
    #print("X_train sample:\n", X_train.head())
    #print("y_train sample:\n", y_train.head())

    # Print the number of churn cases before and after balancing
    print(f"Number of churned donors in training set (after balancing): {sum(y_train == 1)}")
    print(f"Number of non-churned donors in training set (after balancing): {sum(y_train == 0)}")
