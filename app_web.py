import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import gradio as gr

load_dotenv()
API = os.environ.get("OPENROUTER_API_KEY")

# Baselines
BASELINE_LOWER = {
    'Household': 30,
    'Food': 40,
    'Shopping': 7,
    'Transportation': 5,
    'Health & Fitness': 5,
    'Entertainment': 5,
    'Beauty': 4,
    'Investment': 4,
}
BASELINE_UPPER = {
    'Household': 11,
    'Food': 10,
    'Shopping': 13,
    'Transportation': 11,
    'Health & Fitness': 10,
    'Entertainment': 18,
    'Beauty': 8,
    'Investment': 19,
}


# Load model and scaler
def load_financial_model(
    model_path='model/fuzz_dnn_full_model.keras',
    scaler_path='model/fuzzy_dnn_scaler.pkl',
):
    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return None, None


# Prepare features
def prepare_features(df):
    df['spend_deviation_ratio'] = df['Percent_Spend'] / (df['Deviation'].abs() + 1)
    return df[['Percent_Spend', 'Deviation', 'spend_deviation_ratio']]


# Determine income level
def determine_income_level(total_spending):
    return 'upper' if total_spending >= 5000 else 'lower'


# Predict spending pattern
def predict_spending_pattern(model, scaler, input_data):
    total_spending = sum(input_data.values())
    income_level = determine_income_level(total_spending)
    baseline = BASELINE_UPPER if income_level == 'upper' else BASELINE_LOWER

    percent_spend = {k: (v / total_spending) * 100 for k, v in input_data.items()}
    rows = []
    for category, spend_percent in percent_spend.items():
        deviation = spend_percent - baseline.get(category, 0)
        rows.append(
            {
                'Category': category,
                'Percent_Spend': spend_percent,
                'Deviation': deviation,
            }
        )

    pred_df = pd.DataFrame(rows)
    X = prepare_features(pred_df)
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled, verbose=0)

    results = pd.DataFrame(
        {
            'Category': pred_df['Category'],
            'Percent_Spend': pred_df['Percent_Spend'],
            'Deviation': pred_df['Deviation'],
            'Raw_Score': predictions.flatten(),
            'Prediction': ['Good' if pred >= 0.6 else 'Bad' for pred in predictions],
        }
    )
    return (
        results.sort_values('Percent_Spend', ascending=False),
        total_spending,
        income_level,
    )


# Suggest spending pattern
def suggest_spending_pattern(results, total_spending, input_data, income_level):
    results = results.copy()
    suggested_spending = {}
    bad_categories = results[results['Prediction'] == 'Bad']
    good_categories = results[results['Prediction'] == 'Good']
    baseline = BASELINE_UPPER if income_level == 'upper' else BASELINE_LOWER

    if not bad_categories.empty:
        total_to_redistribute = sum(
            input_data[row['Category']]
            * min(max(abs(row['Deviation']) * 0.1, 0.25), 0.50)
            for _, row in bad_categories.iterrows()
            if row['Category'] not in ['Household', 'Food']
        )
        good_total = sum(input_data[cat] for cat in good_categories['Category'])
        distribution_weights = {
            cat: input_data[cat] / good_total if good_total > 0 else 0
            for cat in good_categories['Category']
        }

        for category in input_data:
            original = float(input_data[category])
            baseline_dollars = total_spending * (baseline[category] / 100)
            if category in bad_categories['Category'].values and category not in [
                'Household',
                'Food',
            ]:
                reduction = min(
                    max(
                        abs(
                            results[results['Category'] == category][
                                'Deviation'
                            ].values[0]
                        )
                        * 0.1,
                        0.25,
                    ),
                    0.50,
                )
                suggested = original * (1 - reduction)
            else:
                weight = distribution_weights.get(category, 0)
                increase = total_to_redistribute * weight
                suggested = max(
                    original + increase,
                    baseline_dollars if category in ['Household', 'Food'] else original,
                )
            suggested_spending[category] = (original, round(suggested, 2))
    else:
        suggested_spending = {
            cat: (float(val), float(val)) for cat, val in input_data.items()
        }
    return suggested_spending


# Format for Mistral
def format_for_mistral(
    results, suggested_spending, total_spending, income_level, input_data
):
    return {
        "total_spending": total_spending,
        "income_level": income_level,
        "categories": [
            {
                "category": row['Category'],
                "percent_spend": round(row['Percent_Spend'], 2),
                "actual_dollars": round(input_data[row['Category']], 2),
                "deviation": round(row['Deviation'], 2),
                "prediction": row['Prediction'],
                "suggested_dollars": suggested_spending[row['Category']][1],
            }
            for _, row in results.iterrows()
        ],
    }


# Get spending summary (Mistral API call)
def get_spending_summary(spending_data):
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API)
    analysis_prompt = f"""
    You are a financial counselor analyzing a ${spending_data['total_spending']} monthly budget for a {spending_data['income_level']} income individual. Follow these strict rules:
    ### Financial Literacy Summary
    #### Praise
    For each 'Good' category:
    ⚠️ **Only show if ALL conditions met:**
    - `prediction` = 'Good'
    - `abs(deviation)` < 2%
    ✅ **{{category}} (${{actual_dollars}})** - 
    Explain using:
    1. "% vs baseline: {{percent_spend}}% ({{deviation:+.2f}}% vs {{baseline}}%)"
    2. Practical benefit
    3. Savings impact ONLY if `deviation` > 0
    #### Suggestions 
    ⚠️ **Only show if ALL conditions met:**
    - `prediction` = 'Bad'
    - `abs(deviation)` > 2%
    - `suggested_dollars` ≠ `actual_dollars`
    For each 'Bad' category: 
    ⚠️ **{{category}} (${{actual_dollars}} → ${{suggested_dollars}})** - 
    Structure as:
    1. If suggested INCREASE: "Prioritize {{category}} by adding ${{suggested_dollars - actual_dollars}}..."
    2. If suggested DECREASE: "Reduce {{category}} by ${{actual_dollars - suggested_dollars}}..."
    #### Key Principle
    Identify the MOST URGENT issue using largest absolute deviation...
    **Baseline Reference ({spending_data['income_level'].capitalize()} Income):**
    {'Food (10%), Household (11%), Shopping (13%), Transportation (11%), Health & Fitness (10%), Entertainment (18%), Beauty (8%), Investment (19%)' if spending_data['income_level'] == 'upper' else 'Food (40%), Household (30%), Shopping (7%), Transportation (5%), Health & Fitness (5%), Entertainment (5%), Beauty (4%), Investment (4%)'}
    **Data:**
    {json.dumps(spending_data, indent=2)}
    **Begin Analysis:**
    """
    try:
        response = client.chat.completions.create(
            model="mistralai/mistral-small-24b-instruct-2501:free",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling Mistral API: {e}"


# Gradio interface function
def analyze_spending(
    household,
    food,
    shopping,
    transportation,
    health_fitness,
    entertainment,
    beauty,
    investment,
):
    input_data = {
        'Household': float(household),
        'Food': float(food),
        'Shopping': float(shopping),
        'Transportation': float(transportation),
        'Health & Fitness': float(health_fitness),
        'Entertainment': float(entertainment),
        'Beauty': float(beauty),
        'Investment': float(investment),
    }
    model, scaler = load_financial_model()
    if model is None or scaler is None:
        return "Error: Model or scaler failed to load.", None, None, None

    results, total_spending, income_level = predict_spending_pattern(
        model, scaler, input_data
    )
    suggested_spending = suggest_spending_pattern(
        results, total_spending, input_data, income_level
    )
    spending_data = format_for_mistral(
        results, suggested_spending, total_spending, income_level, input_data
    )
    summary = get_spending_summary(spending_data)

    # Format suggested adjustments as a DataFrame
    suggested_df = pd.DataFrame(
        [(cat, orig, sugg) for cat, (orig, sugg) in suggested_spending.items()],
        columns=['Category', 'Original ($)', 'Suggested ($)'],
    )

    return (
        f"## Spending Analysis ({income_level.capitalize()} Income)\nTotal Spending: ${total_spending:.2f}",
        results,  # For DataFrame display
        suggested_df,  # For DataFrame display
        summary,  # Financial summary
    )


# Gradio UI
with gr.Blocks(
    title="Personal Finance Assistant", css=".gr-button {margin-top: 10px}"
) as demo:
    gr.Markdown("# Personal Finance Assistant")
    gr.Markdown("Enter your monthly spending in each category ($):")
    with gr.Row():
        household = gr.Textbox(label="Household", value="500")
        food = gr.Textbox(label="Food", value="100")
        shopping = gr.Textbox(label="Shopping", value="950")
        transportation = gr.Textbox(label="Transportation", value="100")
    with gr.Row():
        health_fitness = gr.Textbox(label="Health & Fitness", value="200")
        entertainment = gr.Textbox(label="Entertainment", value="200")
        beauty = gr.Textbox(label="Beauty", value="100")
        investment = gr.Textbox(label="Investment", value="100")

    submit_btn = gr.Button("Analyze")

    # Output components
    with gr.Column():
        loading = gr.Markdown("### Analysis Results\n*Waiting for input...*")
        title = gr.Markdown()
        current_spending = gr.DataFrame(label="Current Spending")
        suggested_adjustments = gr.DataFrame(label="Suggested Adjustments")
        financial_summary = gr.Markdown()

    # Handle click with loading state
    def start_loading():
        return "### Analysis Results\n*Processing your spending data...*"

    submit_btn.click(fn=start_loading, inputs=None, outputs=loading).then(
        fn=analyze_spending,
        inputs=[
            household,
            food,
            shopping,
            transportation,
            health_fitness,
            entertainment,
            beauty,
            investment,
        ],
        outputs=[title, current_spending, suggested_adjustments, financial_summary],
        queue=True,
    )

demo.launch()
