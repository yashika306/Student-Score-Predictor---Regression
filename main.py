import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------
# Step 1: Load data
# ---------------------------
df = pd.read_csv("data/StudentsPerformance.csv")

print(df.head())
print(df.shape)
print(df.columns.tolist())
print(df.info())

# ---------------------------
# Step 2: EDA
# ---------------------------
print(df.describe())

categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())

print("\nCorrelation between scores:")
print(df[['math score', 'reading score', 'writing score']].corr())

# ---------------------------
# Step 3: Preprocessing
# ---------------------------
y = df['math score']
X = df.drop(columns=['math score'])
X = pd.get_dummies(X, drop_first=True)

print(X.head())
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# ---------------------------
# Step 4: Train model
# ---------------------------
model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(predictions[:10])
print(y_test[:10].values)

# ---------------------------
# Step 5: Evaluate
# ---------------------------
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f"\nMAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# ---------------------------
# Step 6: Final Report
# ---------------------------
results_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': predictions.round(2),
    'Error': (y_test.values - predictions).round(2)
})

print("\n=== Predicted vs Actual (first 10) ===")
print(results_df.head(10))

print("\n=== Evaluation Report ===")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
print(f"Model explains {r2*100:.1f}% of the variance in math scores.")

# Save results to CSV for reference
results_df.to_csv("data/prediction_results.csv", index=False)
print("\nSaved results to data/prediction_results.csv")