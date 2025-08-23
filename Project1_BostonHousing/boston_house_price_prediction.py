# boston_house_price_prediction.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# -------------------- 1. Load Dataset --------------------
df = pd.read_csv(r"C:\Users\muska\OneDrive\Desktop\g-demo\ShadowFox\HousingData.csv")
print("First 5 rows of dataset:")
print(df.head())

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
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"RMSE: {rmse:.2f}")
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
