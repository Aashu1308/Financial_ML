import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler


def load_model_and_scaler(
    model_path='fuzz_dnn_full_model.keras', scaler_path='fuzzy_dnn_scaler.pkl'
):
    """
    Load the saved model and scaler from disk.
    """
    with open(model_path, 'rb') as model_file:
        model = load_model(model_path)

    with open(scaler_path, 'rb') as scaler_file:
        scaler = joblib.load(scaler_file)

    return model, scaler


def prebuilt_prepare_and_predict(input_data, income_level, model, scaler):
    """
    Prepare the input data, calculate features, scale, and get predictions.

    Parameters:
    - input_data: dict with category-wise spending amounts
    - income_level: 'lower' or 'upper' to select the appropriate baseline
    """

    baseline_upper = {
        'Household': 11,
        'Food': 10,
        'Shopping': 13,
        'Transportation': 11,
        'Health & Fitness': 10,
        'Entertainment': 18,
        'Beauty': 8,
        'Investment': 19,
    }

    baseline_lower = {
        'Household': 30,
        'Food': 40,
        'Shopping': 7,
        'Transportation': 5,
        'Health & Fitness': 5,
        'Entertainment': 5,
        'Beauty': 4,
        'Investment': 4,
    }

    # Select baseline
    baseline = baseline_lower if income_level == 'lower' else baseline_upper

    # Calculate total spending
    total_spending = sum(input_data.values())

    # Convert to percentage spend
    percent_spend = {k: (v / total_spending) * 100 for k, v in input_data.items()}

    # Prepare data for prediction
    rows = []
    for category, spend_percent in percent_spend.items():
        deviation = spend_percent - baseline.get(category, 0)
        row = {
            'Percent_Spend': spend_percent,
            'Deviation': deviation,
            'Category': category,
        }
        rows.append(row)

    # Convert to DataFrame
    pred_df = pd.DataFrame(rows)

    # Create and select interaction features
    pred_df['spend_deviation_ratio'] = pred_df['Percent_Spend'] / (
        pred_df['Deviation'].abs() + 1
    )
    features = ['Percent_Spend', 'Deviation', 'spend_deviation_ratio']

    # Prepare features and scale
    X = pred_df[features]
    X_scaled = scaler.transform(X)

    # Get predictions
    predictions = model.predict(X_scaled)

    # Prepare results
    results = pd.DataFrame(
        {
            'Category': pred_df['Category'],
            'Percent_Spend': pred_df['Percent_Spend'],
            'Deviation': pred_df['Deviation'],
            'Raw_Score': predictions.flatten(),
            'Prediction': ['Good' if pred >= 0.6 else 'Bad' for pred in predictions],
        }
    )

    # Sort by spending percentage
    results = results.sort_values('Percent_Spend', ascending=False)
    results_json = results.to_json(orient='records', lines=True)

    return results_json


# Load model and scaler (just once when starting the app)
model, scaler = load_model_and_scaler()

# Example lower input data
input_data = {
    'Household': 300,
    'Food': 500,
    'Shopping': 200,
    'Transportation': 100,
    'Health & Fitness': 50,
    'Entertainment': 80,
    'Beauty': 30,
    'Investment': 120,
}

# Predict using the model
income_level = 'lower'  # or 'upper'
prediction_results = prebuilt_prepare_and_predict(
    input_data, income_level, model, scaler
)

# Show or return the results in the desired format
print(prediction_results)

# Print summary
# print("\nSummary:")
# print(f"Income category: {income_level}")
# print(
#     f"Categories flagged as Bad: {prediction_results[prediction_results['Prediction'] == 'Bad']['Category'].tolist()}"
# )
# print(
#     f"Highest spending category: {prediction_results.iloc[0]['Category']} "
#     f"({prediction_results.iloc[0]['Percent_Spend']:.1f}%)"


# Example upper input data

input_data = {
    'Household': 900,
    'Food': 1500,
    'Shopping': 1300,
    'Transportation': 1100,
    'Health & Fitness': 1000,
    'Entertainment': 1500,
    'Beauty': 800,
    'Investment': 2100,
}

# Predict using the model
income_level = 'upper'  # or 'upper'
prediction_results = prebuilt_prepare_and_predict(
    input_data, income_level, model, scaler
)

# Show or return the results in the desired format
print(prediction_results)

# Print summary
# print("\nSummary:")
# print(f"Income category: {income_level}")
# print(
#     f"Categories flagged as Bad: {prediction_results[prediction_results['Prediction'] == 'Bad']['Category'].tolist()}"
# )
# print(
#     f"Highest spending category: {prediction_results.iloc[0]['Category']} "
#     f"({prediction_results.iloc[0]['Percent_Spend']:.1f}%)"
# )
