import pandas as pd
import numpy as np

df = pd.read_csv("dataset/lower_income.csv")
df2 = pd.read_csv("dataset/upper_income.csv")

# Convert 'Date' to datetime if not already done
df['Date'] = pd.to_datetime(df['Date'])

# Check for empty 'Income/Expense' values
empty_rows = df[df['Type'].isna()]


# Set type to 'Income' for 'Salary' and assign random date between first and last entry
def assign_income_expense(row):
    if pd.isna(row['Type']):  # Check if the Income/Expense value is empty
        if row['Category'] == 'Salary':
            # Set 'Income' type
            row['Type'] = 'Income'
            # Generate a random date between the first and last date in the dataset
            if pd.isna(row['Date']):
                random_date = pd.to_datetime(
                    np.random.choice(pd.date_range(df['Date'].min(), df['Date'].max()))
                )
                row['Date'] = random_date
        else:
            # Set 'Expense' type for other categories
            row['Type'] = 'Expense'
            if pd.isna(row['Date']):
                random_date = pd.to_datetime(
                    np.random.choice(pd.date_range(df['Date'].min(), df['Date'].max()))
                )
                row['Date'] = random_date
    return row


def update_row(row):
    if row['Category'] == 'Food':
        row['Type'] = 'Expense'
    return row


# Apply the function to rows with missing 'Income/Expense' values
# df = df.apply(assign_income_expense, axis=1)
# df2 = df2.apply(update_row, axis=1)
df = df.apply(update_row, axis=1)

# Print the updated DataFrame to verify
# print(df[['Category', 'Type', 'Date']].head())
# f = True
# for column in df.columns:
#     if df[column].isna().any():
#         print(f"Column {column} has {df[column].isna().sum()} missing values")
#         f = False
# if f:
#     print("No empty columns")
# print(pd.isna(df['Date']))
df.to_csv("dataset/lower_income.csv", index=False)
# df2.to_csv("dataset/upper_income.csv", index=False)
