import pandas as pd

# Load the CSV files into DataFrames
df1 = pd.read_csv('dataset/lower_income.csv')
df2 = pd.read_csv('dataset/lower2.csv')

# Display the shapes of the DataFrames
print(f"Shape of file1: {df1.shape}")
print(f"Shape of file2: {df2.shape}")

# Check if the DataFrames are equal
if df1.equals(df2):
    print("The two DataFrames are identical.")
else:
    print("The two DataFrames are different.")

# Find differences
# 1. Check for missing values in either DataFrame
missing_in_df1 = df1.isnull().sum()
missing_in_df2 = df2.isnull().sum()

print("\nMissing values in file1:")
print(missing_in_df1[missing_in_df1 > 0])

print("\nMissing values in file2:")
print(missing_in_df2[missing_in_df2 > 0])

# 2. Find rows that are different
# Merge the two DataFrames to find differences
comparison_df = df1.compare(df2)

print("\nDifferences between the two DataFrames:")
print(comparison_df)

# 3. Find rows that are in one DataFrame but not the other
only_in_df1 = df1[~df1.apply(tuple, 1).isin(df2.apply(tuple, 1))]
only_in_df2 = df2[~df2.apply(tuple, 1).isin(df1.apply(tuple, 1))]

print("\nRows only in file1:")
print(only_in_df1)

print("\nRows only in file2:")
print(only_in_df2)
