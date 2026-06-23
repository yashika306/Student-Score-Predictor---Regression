import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------
# Step 1: Load data
# ---------------------------
df = pd.read_csv("data/StudentsPerformance.csv")

print("=== Data Preview ===")
print(df.head())
print(df.shape)
print(df.isnull().sum())

# ---------------------------
# Step 2: EDA
# ---------------------------
print("\n=== Summary Statistics ===")
print(df.describe())

categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())

print("\n=== Correlation Between Scores ===")
print(df[['math score', 'reading score', 'writing score']].corr())

# ---------------------------
# Step 3: Preprocessing (done ONCE, shared by all models)
# ---------------------------
y = df['math score']
X = pd.get_dummies(df.drop(columns=['math score']), drop_first=True)

print("\n=== Preprocessed Features ===")
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train size:", X_train.shape, "| Test size:", X_test.shape)

# ---------------------------
# Step 4: Train + Evaluate ALL models in one loop (each trained ONCE)
# ---------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
}

results = []
predictions_by_model = {}

for name, model in models.items():
    model.fit(X_train, y_train)              # train ONCE
    preds = model.predict(X_test)             # predict ONCE
    predictions_by_model[name] = preds        # store for later use

    results.append({
        "Model": name,
        "MAE": round(mean_absolute_error(y_test, preds), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 2),
        "R2": round(r2_score(y_test, preds), 4),
    })

comparison_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)

print("\n=== Model Comparison ===")
print(comparison_df)

# ---------------------------
# Step 5: Build one CSV with Actual + every model's Predicted + Error
# ---------------------------
results_df = pd.DataFrame({'Actual': y_test.values})

for name, preds in predictions_by_model.items():
    col_name = name.replace(" ", "_")  # e.g. "Linear_Regression"
    results_df[f'{col_name}_Predicted'] = preds.round(2)
    results_df[f'{col_name}_Error'] = (y_test.values - preds).round(2)

print("\n=== Predicted vs Actual, all models (first 10) ===")
print(results_df.head(10))

best_model_name = comparison_df.iloc[0]["Model"]
print(f"\nBest model: {best_model_name}")

results_df.to_csv("data/prediction_results.csv", index=False)
print("\nSaved all model predictions to data/prediction_results.csv")

comparison_df.to_csv("data/model_comparison.csv", index=False)
print("Saved model comparison metrics to data/model_comparison.csv")