
import pickle
import os
import sys

# --- Configuration ---
MODEL_FILE = os.path.join('./custom_forecasting_model/output', 'linear_regression_model.pkl')

# --- Load the Model ---
print(f"Loading model from {MODEL_FILE}...")
try:
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_FILE}. Please run the training script first.")
    sys.exit(1)

m = model['m']
c = model['c']
print("Model loaded successfully.")

# --- Make a Prediction ---
if __name__ == '__main__':
    if len(sys.argv) > 1:
        try:
            year_to_predict = int(sys.argv[1])
            prediction = m * year_to_predict + c
            print(f"\nPrediction for year {year_to_predict}: {prediction:.2f}")
        except ValueError:
            print("Error: Please provide a valid year as a command-line argument.")
    else:
        print("\nUsage: python predict.py <year>")
        # Example prediction
        example_year = 2030
        prediction = m * example_year + c
        print(f"Example prediction for year {example_year}: {prediction:.2f}")
