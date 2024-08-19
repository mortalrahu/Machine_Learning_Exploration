import requests
import json

# Define the URL of the Flask API
url = 'http://127.0.0.1:5000/predict'

# Corrected data format with original column names
data = {
    'age': 50,
    'donation_amount': 100.0,
    'donation_frequency': 'monthly',  # e.g., 'monthly', 'quarterly', 'yearly'
    'years_as_donor': 8,
    'recent_donations': 'yes'  # e.g., 'yes', 'no'
}

# Send a POST request to the API
response = requests.post(url, json=data)

# Print the response from the API
print("Response from API:", response.json())
