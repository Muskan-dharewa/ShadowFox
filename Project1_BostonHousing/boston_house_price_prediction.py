# boston_house_price_prediction.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# -------------------- 1. Load Dataset --------------------
# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the path to the CSV file
csv_path = os.path.join(script_dir, "HousingData.csv")
print("Loading CSV from:", csv_path)

# Load dataset
df = pd.read_csv(csv_path)
print("First 5 rows of dataset:")
print(df.head())

# Optional: check for correlation
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.show()

# -------------------- 2. Data Preprocessing --------------------
# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Separate features (X) and target (y)
X = df.drop("MEDV", axis=1)  # MEDV = Median value of owner-occupied homes
y = df["MEDV"]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------- 3. Train-Test Split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------- 4. Model Selection & Training --------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------- 5. Predictions --------------------
y_pred = model.predict(X_test)

# -------------------- 6. Evaluation --------------------
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# -------------------- 7. Enhanced Visualization --------------------
plt.figure(figsize=(12, 5))

# (A) Actual vs Predicted
plt.subplot(1, 2, 1)
sns.regplot(x=y_test, y=y_pred,
            scatter_kws={"alpha": 0.6, "color": "skyblue"},
            line_kws={"color": "red", "lw": 2})
plt.title("Actual vs Predicted Prices", fontsize=14, fontweight="bold")
plt.xlabel("Actual Prices (MEDV)")
plt.ylabel("Predicted Prices")
plt.grid(True, linestyle='--', alpha=0.7)

# (B) Error Distribution
plt.subplot(1, 2, 2)
errors = y_test - y_pred
sns.histplot(errors, kde=True, color="purple", alpha=0.6)
plt.title("Prediction Error Distribution", fontsize=14, fontweight="bold")
plt.xlabel("Error (Actual - Predicted)")
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# -------------------- 8. Additional Insights --------------------
# Create a DataFrame for actual and predicted values
pred_df = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": y_pred
})

# Sort by predicted prices
pred_df_sorted = pred_df.sort_values(by="Predicted Price").reset_index(drop=True)

print("\nHouses ranked from lowest to highest predicted price:")
print(pred_df_sorted.head(10))  # Show top 10 as example

print("\nTop 5 most expensive houses (predicted):")
print(pred_df_sorted.tail(5))

print("\nBottom 5 cheapest houses (predicted):")
print(pred_df_sorted.head(5))

# Line plot of predicted prices distribution
plt.figure(figsize=(10, 4))
plt.plot(range(len(pred_df_sorted)), pred_df_sorted["Predicted Price"], marker='o', linestyle='-',
         color="green", alpha=0.7)
plt.title("Predicted House Prices Ranked from Lowest to Highest", fontsize=14, fontweight="bold")
plt.xlabel("House Index (Sorted by Predicted Price)")
plt.ylabel("Predicted Price")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
