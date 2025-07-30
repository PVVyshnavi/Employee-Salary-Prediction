# Employee-Salary-Prediction
Project Objective (Theory)
The goal of this project is to build a Machine Learning model that can predict an employee’s salary based on key features such as:

Age

Gender

Education Level

Job Title

Years of Experience

By analyzing historical employee data, we aim to train a regression model that learns the relationship between these features and salary, and then use it to predict future salaries.

🧪 Technologies & Tools Used
Programming Language: Python

Libraries Used:

pandas – for data loading & processing

matplotlib & seaborn – for data visualization

scikit-learn – for model building (Linear Regression, Encoding, Metrics)

🏗️ Step-by-Step Implementation (Theory)
Import Required Libraries

Load the Dataset

Load data using pandas.read_csv().

Explore the Data

Display sample rows and info() of the dataset.

Data Preprocessing

Handle missing values using dropna().

Encode categorical variables using LabelEncoder from sklearn.

Feature Selection

Features (X): Age, Gender, Education Level, Job Title, Years of Experience

Target (y): Salary

Split the Dataset

Use train_test_split() to split into 80% training and 20% testing.

Model Training

Use LinearRegression() from scikit-learn to fit the model on training data.

Model Evaluation

Evaluate using:

Mean Squared Error (MSE)

R² Score (R-squared)

Visualization

Histogram of Salary

Scatter Plot of Experience vs Salary

Output

Save predictions to salary_predictions.csv.

💻 Installation Commands
Make sure Python 3.x is installed. Then run:

bash
Copy
Edit
# Create virtual environment (optional but recommended)
python -m venv venv
# Activate the environment:
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required libraries
pip install pandas matplotlib seaborn scikit-learn
🚀 Run the Project
bash
Copy
Edit
python main.py
This will:

Print data info

Train the model

Show MSE and R²

Save visualizations as PNG

Output predictions to salary_predictions.csv

✅ Summary
You implemented a complete machine learning pipeline.

You applied data cleaning, feature encoding, regression modeling, and visualizations.

The model performance was evaluated with MSE and R² Score.
