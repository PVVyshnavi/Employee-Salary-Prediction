import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("salary_data.csv")

# Clean data
df.dropna(inplace=True)

print("First 5 rows of dataset:\n", df.head(), "\n")
print("Dataset Info:\n", df.info(), "\n")

# ----------------- 📊 Visualizations -----------------

# Salary distribution
plt.figure(figsize=(6, 4))
sns.histplot(df['Salary'], bins=20, kde=True)
plt.title("Salary Distribution")
plt.savefig("salary_distribution.png")
plt.close()

# Years of Experience vs Salary
plt.figure(figsize=(6, 4))
sns.scatterplot(x='Years of Experience', y='Salary', data=df)
plt.title("Experience vs Salary")
plt.savefig("experience_vs_salary.png")
plt.close()

# Correlation heatmap (numerical only)
plt.figure(figsize=(6, 4))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="YlGnBu")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

# ----------------- 🎯 Model Training -----------------

# Features and Target
X = df.drop("Salary", axis=1)
y = df["Salary"]

# ColumnTransformer for categorical encoding
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

# Pipeline
model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('regressor', LinearRegression())
])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# ----------------- 📈 Prediction vs Actual Plot -----------------
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.savefig("prediction_vs_actual.png")
plt.close()

# Save predictions to CSV
pred_df = pd.DataFrame({'Actual Salary': y_test, 'Predicted Salary': y_pred})
pred_df.to_csv("salary_predictions.csv", index=False)

print("Graphs saved. Predictions saved in 'salary_predictions.csv'")
