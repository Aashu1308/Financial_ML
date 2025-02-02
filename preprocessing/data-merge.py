import pandas as pd

# Load the CSV files into DataFrames
df1 = pd.read_csv('dataset/lower_income1.csv')
df2 = pd.read_csv('dataset/upper_income1.csv')


def update_row_1(row):
    if row['Category'] == 'Self-development':
        row['Category'] = 'Health & Fitness'
    if row['Category'] == 'Apparel':
        row['Category'] = 'Shopping'
    if row['Category'] == 'Social Life':
        row['Category'] = 'Entertainment'
    if row['Category'] == 'Other':
        row['Category'] = 'Investment'
    return row


def update_row_2(row):
    if row['Category'] == 'Food & Drink':
        row['Category'] = 'Food'
    if row['Category'] == 'Travel':
        row['Category'] = 'Transportation'
    if row['Category'] == 'Other':
        row['Category'] = 'Education'
    if row['Category'] == 'Utilities':
        row['Category'] = 'Beauty'
    if row['Category'] == 'Rent':
        row['Category'] = 'Household'
    return row


# df1 = df1.apply(update_row_1, axis=1)
# df2 = df2.apply(update_row_2, axis=1)

# cat1 = df1['Category']
# cat2 = df2['Category']

# for i in zip(cat1.unique(), cat2.unique()):
#     print(i)

# df1.to_csv("dataset/lower_income1.csv", index=False)
# df2.to_csv("dataset/upper_income1.csv", index=False)

avg_lower = df1.groupby('Category')['Amount'].mean().sort_values(ascending=False)
avg_upper = df2.groupby('Category')['Amount'].mean().sort_values(ascending=False)

print("Lower")
print(avg_lower)
print("Upper")
print(avg_upper)

# Calculate total spend for each dataset
total_spend_lower = df1['Amount'].sum()
total_spend_upper = df2['Amount'].sum()

# Calculate percentage spend for each category
percent_lower = (
    df1.groupby('Category')['Amount'].sum() / total_spend_lower * 100
).sort_values(ascending=False)
percent_upper = (
    df2.groupby('Category')['Amount'].sum() / total_spend_upper * 100
).sort_values(ascending=False)

# Display the results
print("Percentage of Spending by Category (Lower Income):")
print(percent_lower)

print("\nPercentage of Spending by Category (Upper Income):")
print(percent_upper)


# Function to swap 'Food' and 'Investment' labels in df1
def swap_food_investment(row):
    if row['Category'] == 'Food':
        row['Category'] = 'Investment_temp'
    elif row['Category'] == 'Investment':
        row['Category'] = 'Food'
    elif row['Category'] == 'Investment_temp':
        row['Category'] = 'Investment'
    return row


# Function to swap 'Food' and 'Education' labels in df1
def swap_edu(row):
    if row['Category'] == 'Food':
        row['Category'] = 'Edu_temp'
    elif row['Category'] == 'Education':
        row['Category'] = 'Food'
    elif row['Category'] == 'Edu_temp':
        row['Category'] = 'Food'
    return row


def finalise_row(row):
    if row['Category'] == 'Edu_temp':
        row['Category'] = 'Food'
    return row


# Apply the swap function
df1 = df1.apply(swap_edu, axis=1)
df1 = df1.apply(finalise_row, axis=1)


# Recalculate averages and percentages after swapping
avg_lower = df1.groupby('Category')['Amount'].mean().sort_values(ascending=False)
total_spend_lower = df1['Amount'].sum()
percent_lower = (
    df1.groupby('Category')['Amount'].sum() / total_spend_lower * 100
).sort_values(ascending=False)

# Display results
print("Updated Lower Income Averages:")
print(avg_lower)
print("\nUpdated Percentage of Spending by Category (Lower Income):")
print(percent_lower)

df1.to_csv("dataset/lower_income1.csv", index=False)
# df2.to_csv("dataset/upper_income1.csv", index=False)
