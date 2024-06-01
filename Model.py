import mysql.connector
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# Step 1: Fetch data from the database
# Connect to MySQL
db_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="karthi",
    database="elevator_predictor"
)

# Fetch data from MainData and SensorX tables
cursor = db_connection.cursor()

cursor.execute("SELECT M.revolutions, M.humidity, S.x1, S.x2, S.x3, S.x4, S.x5, M.vibration FROM MainData M INNER JOIN SensorX S ON M.ID = S.ID;")
data = cursor.fetchall()

# Close cursor and database connection
cursor.close()
db_connection.close()

# Step 2: Prepare data
# Split data into features and target variable (vibration)
features = []
labels = []
for row in data:
    features.append(list(row[:-1]))  # Convert tuple to list and exclude vibration (last column)
    labels.append(row[-1])           # Last column is vibration

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Step 3: Model building
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Evaluation
# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)

# Step 5: Prediction
# Define new input values for prediction
new_input_data = np.array([[93.744, 73.999, 167.743, 19.745, 1.26682793, 8787.937536, 5475.852001]])

# Make predictions
predictions = model.predict(new_input_data)

print("Predicted vibration:", predictions)

# Step 6: Save the model
joblib.dump(model, "elevator_predictor_model_linear_regression.pkl")
