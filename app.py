import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()
API = os.environ.get("OPENROUTER_API_KEY")


# Load model and scaler
def load_financial_model(
    model_path='model/fuzz_dnn_full_model.keras',
    scaler_path='model/fuzzy_dnn_scaler.pkl',
):
    """Load the pre-trained model and scaler from files."""
    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        print("Model and scaler loaded successfully")
        return model, scaler
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return None, None


# Prepare features
def prepare_features(df):
    df['spend_deviation_ratio'] = df['Percent_Spend'] / (df['Deviation'].abs() + 1)
    return df[['Percent_Spend', 'Deviation', 'spend_deviation_ratio']]


def predict_spending_pattern(model, scaler, input_data, income_level='lower'):
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
    baseline = baseline_lower if income_level == 'lower' else None
    total_spending = sum(input_data.values())
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
    return results.sort_values('Percent_Spend', ascending=False), total_spending


def suggest_spending_pattern(results, total_spending, input_data):
    results = results.copy()
    suggested_spending = {}
    bad_categories = results[results['Prediction'] == 'Bad']
    good_categories = results[results['Prediction'] == 'Good']
    baseline = {
        'Household': 30,
        'Food': 40,
        'Shopping': 7,
        'Transportation': 5,
        'Health & Fitness': 5,
        'Entertainment': 5,
        'Beauty': 4,
        'Investment': 4,
    }

    if not bad_categories.empty:
        # Only reduce non-essential "Bad" categories
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


def format_for_mistral(results, suggested_spending, total_spending):
    return {
        "total_spending": total_spending,
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


def get_spending_summary(spending_data):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=API,
    )

    analysis_prompt = f"""
You are a financial counselor analyzing a ${spending_data['total_spending']} monthly budget. Follow these strict rules:

### Financial Literacy Summary

#### Praise
For each 'Good' category:
⚠️ **Only show if ALL conditions met:**
- `prediction` = 'Good'
- `abs(deviation)` < 2%

✅ **{{category}} (${{actual_dollars}})** - 
Explain using:
1. "% vs baseline: {{percent_spend}}% ({{deviation:+.2f}}% vs {{baseline}}%)"
2. Practical benefit ("maintains nutrition", "ensures reliable transportation")
3. Savings impact ONLY if only if `deviation` > 0

#### Suggestions 
⚠️ **Only show if ALL conditions met:**
- `prediction` = 'Bad'
- `abs(deviation)` > 2%
- `suggested_dollars` ≠ `actual_dollars`

For each 'Bad' category: 
⚠️ **{{category}} (${{actual_dollars}} → ${{suggested_dollars}})** - 
Structure as:
1. If suggested INCREASE (suggested_dollars > actual_dollars):
   - "Prioritize {{category}} by adding ${{suggested_dollars - actual_dollars}} from other areas to meet {{baseline}}% baseline"
   - "Needed for: [specific essentials]"
2. If suggested DECREASE (suggested_dollars < actual_dollars):
   - "Reduce {{category}} by ${{actual_dollars - suggested_dollars}} ({{percent_spend}}% → {{suggested_dollars/total_spending*100:.1f}}%)"
   - "Freed funds could: [specific use]"

#### Key Principle
Identify the MOST URGENT issue using:
1. Largest absolute deviation as max_deviation (use `abs(deviation)`)
{{% if max_deviation > 10 %}}
"Critical priority: Address {{max_dev_category}} ({{max_dev:+.2f}}% deviation)"
{{% elif max_deviation > 5 %}}
"Important adjustment: Improve {{max_dev_category}} management"
{{% else %}}
"Maintain current allocations - all categories within acceptable variance"
{{% endif %}}
2. Practical consequence of current spending
3. Specific financial rule ("50/30/20", "pay yourself first") and how it can be applied in current scenario

**Hard Rules:**
- NEVER use "freeing" for suggested increases
- For increase: "Reallocate ${{suggested_dollars - actual_dollars}} from lower-priority categories"
- For decrease: "Recover ${{actual_dollars - suggested_dollars}} for core needs"
- Deviation signs MUST match: "+" for overspend, "-" for underspend
- max_deviation MUST be calculated based on absolute values of deviation percentages (`abs(deviation)`)
- Never suggest changes for 'Good' predictions
- Never show Suggestions section if no valid 'Bad' categories
- Use exact numbers from data - no calculations
- Deviation thresholds:
   - Minor: ±2-5%
   - Significant: ±5-10%
   - Critical: >±10%


**Baseline Reference:**
Food (40%), Household (30%), Beauty (4%), Entertainment (5%), Shopping (7%), Transportation (5%), Health (5%), Investments (4%)

**Data:**
{json.dumps(spending_data, indent=2)}

**Example Output:**
### Financial Literacy Summary

#### Praise
✅ **Food ($800)** - 
- % vs baseline: 39.02% (-0.98% vs 40%)
- Practical benefit: Maintains nutrition and health
- Savings impact: N/A

✅ **Entertainment ($200)** - 
- % vs baseline: 9.76% (+4.76% vs 5%)
- Practical benefit: Ensures relaxation and social interaction
- Savings impact: Recover $97.56 for core needs if reduced to baseline

✅ **Shopping ($150)** - 
- % vs baseline: 7.32% (+0.32% vs 7%)
- Practical benefit: Ensures necessary personal items
- Savings impact: Recover $6.56 for core needs if reduced to baseline

✅ **Transportation ($100)** - 
- % vs baseline: 4.88% (-0.12% vs 5%)
- Practical benefit: Ensures reliable transportation
- Savings impact: N/A

✅ **Health & Fitness ($100)** - 
- % vs baseline: 4.88% (-0.12% vs 5%)
- Practical benefit: Maintains physical well-being
- Savings impact: N/A

#### Suggestions
⚠️ **Household ($400 → $615)** - 
- Prioritize Household by adding $215 from other areas to meet 30% baseline
- Needed for: Rent, utilities, and home maintenance

⚠️ **Beauty ($300 → $150)** - 
- Reduce Beauty by $150 (14.63% → 7.3%)
- Freed funds could: Cover 2.9 months of baseline Transportation costs

#### Key Principle
The MOST URGENT issue is the **Beauty** category.
- Excessive spending in Beauty compromises core needs like rent, utilities, and maintenance.

#### Recommendations
1. **Immediate Action**: Reduce Beauty expenses by $150 to meet the baseline of 4% spending
2. **Long-Term Strategy**: Reallocate $215 from lower-priority categories to meet the 30% baseline for Household expenses.


**Begin Analysis:**
"""

    try:
        response = client.chat.completions.create(
            model="mistralai/mistral-small-24b-instruct-2501:free",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.5,  # Optional: reduces variance
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling Mistral API: {e}")
        return "Unable to generate summary due to API error."


def main():
    global input_data
    input_data = {
        'Household': 500,
        'Food': 100,
        'Shopping': 950,
        'Transportation': 100,
        'Health & Fitness': 200,
        'Entertainment': 200,
        'Beauty': 100,
        'Investment': 100,
    }
    model, scaler = load_financial_model()
    if model is None or scaler is None:
        return

    results, total_spending = predict_spending_pattern(
        model, scaler, input_data, 'lower'
    )
    suggested_spending = suggest_spending_pattern(results, total_spending, input_data)
    spending_data = format_for_mistral(results, suggested_spending, total_spending)
    summary = get_spending_summary(spending_data)

    print(f"\nSpending Analysis (Lower Income):")
    print(results.to_string(index=False))
    print(f"\nTotal Spending: ${total_spending:.2f}")
    print("\nSuggested Adjustments:")
    for cat, (orig, sugg) in suggested_spending.items():
        print(f"{cat}: ${orig} → ${sugg}")
    print("\nFinancial Literacy Summary from Mistral Small:")
    print(summary)


if __name__ == "__main__":
    main()
