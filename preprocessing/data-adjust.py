import pandas as pd

# Load the CSV file into a DataFrame
df2 = pd.read_csv('dataset/upper_income1.csv')

# Filter out rows where Category is 'Education'
# df2 = df2[df2['Category'] != 'Education']

# Multiply the Amount by 1.5 for Entertainment and Shopping categories
# df2.loc[df2['Category'].isin(['Entertainment', 'Shopping']), 'Amount'] *= 1.5
df2.loc[df2['Category'].isin(['Entertainment']), 'Amount'] *= 1.5

# Save the updated DataFrame back to the CSV file
df2.to_csv('dataset/upper_income1.csv', index=False)

print(
    "Rows with 'Entertaiment' in Category have been increased and the file is updated."
)
