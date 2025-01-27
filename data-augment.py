import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Load and clean the lower income dataset
df = pd.read_csv("dataset/expense_data_1.csv")
df = df.drop(['Account', 'Subcategory', 'Note', 'INR', 'Note.1', 'Currency'], axis=1)
df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

# Encode the 'Category' column
le = LabelEncoder()
df['Category_encoded'] = le.fit_transform(df['Category'])

# Count occurrences of each category
category_counts = df['Category'].value_counts()

# Filter categories with more than 1 sample
filtered_categories = category_counts[category_counts > 1].index
df_filtered = df[df['Category'].isin(filtered_categories)]

# Save columns
income = df_filtered['Income/Expense']
date = df_filtered['Date']

# Proceed with SMOTE on the filtered dataset
X = df_filtered[['Amount']]
y = df_filtered['Category_encoded']

# Apply SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=1)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Convert back to DataFrame
df_smote = pd.DataFrame(X_resampled, columns=['Amount'])
df_smote['Category_encoded'] = y_resampled
df_smote['Category'] = le.inverse_transform(df_smote['Category_encoded'])
df_smote['Type'] = income
df_smote['Date'] = date
df_smote = df_smote.drop(['Category_encoded'], axis=1)


# Assuming df_smote is already created and contains the oversampled data
categories_to_downsample_heavily = [
    'Apparel',
    'Household',
    'Social Life',
    'Education',
    'Self-development',
    'Beauty',
]

# Separate the categories to be downsampled and the rest
df_to_downsample = df_smote[df_smote['Category'].isin(categories_to_downsample_heavily)]
df_remaining = df_smote[~df_smote['Category'].isin(categories_to_downsample_heavily)]

# Downsample each category to a random count within the range of 35 to 80
df_downsampled_list = []
for category in categories_to_downsample_heavily:
    df_category = df_to_downsample[df_to_downsample['Category'] == category]
    random_count = np.random.randint(50, 100)  # Random count between 35 and 80
    df_downsampled = resample(
        df_category,
        replace=False,
        n_samples=min(len(df_category), random_count),
        random_state=42,
    )
    df_downsampled_list.append(df_downsampled)

# Combine the downsampled categories and the remaining data
df_downsampled = pd.concat(df_downsampled_list)
df_combined = pd.concat([df_downsampled, df_remaining])

# Rename 'Allowance' category to 'Salary' in df_combined
df_combined['Category'] = df_combined['Category'].replace('Allowance', 'Salary')

# Assuming df_combined is already created and contains the oversampled data
categories_to_downsample_slightly = [
    'Food',
    'Other',
    'Transportation',
]

# Separate the categories to be downsampled and the rest
df_to_downsample = df_combined[
    df_combined['Category'].isin(categories_to_downsample_slightly)
]
df_remaining = df_combined[
    ~df_combined['Category'].isin(categories_to_downsample_slightly)
]

# Downsample each category to a random count within the range of 35 to 80
df_downsampled_list = []
for category in categories_to_downsample_slightly:
    df_category = df_to_downsample[df_to_downsample['Category'] == category]
    random_count = np.random.randint(98, 150)  # Random count between 35 and 80
    df_downsampled = resample(
        df_category,
        replace=False,
        n_samples=min(len(df_category), random_count),
        random_state=42,
    )
    df_downsampled_list.append(df_downsampled)

# Combine the downsampled categories and the remaining data
df_downsampled = pd.concat(df_downsampled_list)
df_final = pd.concat([df_downsampled, df_remaining])
# df_final.drop(columns=df_final.columns[0], axis=1, inplace=True)

# Display the result
print("Updated Lower Income Spend Categories with Random Downsampling")
print(df_final['Category'].value_counts())


# # Display the result
# print("Lower income spend")
# print(df_smote['Category'].value_counts())

# print("Prev spend")
# print(df['Category'].value_counts())

# Load and clean the higher income dataset
df2 = pd.read_csv("dataset/Personal_Finance_Dataset.csv")
df2 = df2.drop(['Transaction Description'], axis=1)
# df2['Amount']   # Normalize Amount
df2['Date'] = pd.to_datetime(df2['Date']).dt.strftime('%Y-%m-%d')

# Increase salary
df2.loc[df2['Category'] == 'Salary', 'Amount'] *= 3

print("Higher income spend categories")
print(df2['Category'].value_counts())

# Group by Category and calculate the average of Amount
avg_combined = df_final.groupby('Category')['Amount'].mean()

# Display the result
print("Average Amount per Category in lower income:")
print(avg_combined)

# Group by Category and calculate the average of Amount
avg_df2 = df2.groupby('Category')['Amount'].mean()

# Display the result
print("Average Amount per Category in higher income:")
print(avg_df2)

df_final.to_csv("dataset/lower_income.csv", index=False)
df2.to_csv("dataset/upper_income.csv", index=False)
