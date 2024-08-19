import numpy as np
import pandas as pd

class ChurnPredictionDataGenerator:
    """
    Class to generate a churn prediction dataset for a charity organization.
    """

    def __init__(self, num_samples=1000, churn_ratio=0.2):
        """
        Initialize the generator with the number of samples and churn ratio.

        Args:
            num_samples (int): The number of donor samples to generate.
            churn_ratio (float): The proportion of churned donors (e.g., 0.2 for 20% churn).
        """
        self.num_samples = num_samples
        self.churn_ratio = churn_ratio

    def generate_data(self):
        """
        Generate a churn prediction dataset with the given characteristics.

        Returns:
            pd.DataFrame: The generated dataset as a pandas DataFrame.
        """
        try:
            # Set random seed for reproducibility
            np.random.seed(42)

            # Generate donor attributes
            donor_ids = np.arange(1, self.num_samples + 1)
            ages = np.random.randint(18, 80, size=self.num_samples)
            donation_amounts = np.random.choice([10, 25, 50, 100, 200, 500], size=self.num_samples)
            donation_frequency = np.random.choice(['monthly', 'quarterly', 'yearly'], size=self.num_samples)
            years_as_donor = np.random.randint(1, 10, size=self.num_samples)
            recent_donations = np.random.choice(['yes', 'no'], size=self.num_samples, p=[0.6, 0.4])

            # Generate churn labels with an imbalance (more non-churn than churn)
            churn_labels = np.random.choice([1, 0], size=self.num_samples, p=[self.churn_ratio, 1 - self.churn_ratio])

            # Combine the generated data into a DataFrame
            churn_data = pd.DataFrame({
                'donor_id': donor_ids,
                'age': ages,
                'donation_amount': donation_amounts,
                'donation_frequency': donation_frequency,
                'years_as_donor': years_as_donor,
                'recent_donations': recent_donations,
                'churn': churn_labels
            })

            return churn_data

        except Exception as e:
            print(f"Error occurred while generating data: {e}")
            return pd.DataFrame()  # Return empty DataFrame in case of an error


if __name__ == "__main__":
    generator = ChurnPredictionDataGenerator(num_samples=1000, churn_ratio=0.2)
    churn_data = generator.generate_data()

    churn_data.to_csv('churn_data.csv', index=False)

    # Display first few rows of the generated data
    print(churn_data.head())
