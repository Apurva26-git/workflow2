import os
import pandas as pd

# Load CSV file
df = pd.read_csv('ModelCleaning/data.csv')

print("Original Data:")
print(df.head())

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Remove leading/trailing whitespace and lowercase string values
df = df.applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)

# Drop rows with any missing values
df.dropna(inplace=True)

# Drop duplicate rows
df.drop_duplicates(inplace=True)

workspace = os.getenv('GITHUB_WORKSPACE')
model_cleaning_dir = os.path.join(workspace, 'ModelCleaning')
os.makedirs(model_cleaning_dir, exist_ok=True)

output_path = os.path.join(model_cleaning_dir, 'data.csv')
df.to_csv(output_path, index=False)

print("Cleaned Data:")
print(df.head())
print(f"\nCleaned data saved to '{output_path}'")
