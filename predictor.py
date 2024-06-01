import numpy as np
import joblib

# Load the trained model
model = joblib.load(r"C:/Users/karth/Downloads/Elevator_Predictor/elevator_predictor_model_linear_regression.pkl")

# Get input from the user
revolutions = float(input("Enter revolutions: "))
humidity = float(input("Enter humidity: "))
x1 = float(input("Enter x1: "))
x2 = float(input("Enter x2: "))
x3 = float(input("Enter x3: "))
x4 = float(input("Enter x4: "))
x5 = float(input("Enter x5: "))

# Create input array for prediction
new_input_data = np.array([[revolutions, humidity, x1, x2, x3, x4, x5]])

# Make predictions
predictions = model.predict(new_input_data)

print("Predicted vibration:", predictions[0])
