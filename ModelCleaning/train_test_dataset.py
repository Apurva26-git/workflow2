import os
from pandas import read_csv
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

workspace = os.getenv('GITHUB_WORKSPACE')
model_cleaning_dir = os.path.join(workspace, 'ModelCleaning')
/workspaces/workflow2/ModelCleaning/data.csv = os.path.join(model_cleaning_dir, 'data.csv')

if os.path.exists(/workspaces/workflow2/ModelCleaning/data.csv):
    print(f"File found: {/workspaces/workflow2/ModelCleaning/data.csv}")
else:
    print(f"File not found at: {/workspaces/workflow2/ModelCleaning/data.csv}")

df = read_csv(/workspaces/workflow2/ModelCleaning/data.csv)
print(df.head())

X = df["Age"].values.reshape(-1, 1)
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

dump(model, os.path.join(model_cleaning_dir, "AgeSalaryModel.pkl"))
print("Model trained and saved.")
