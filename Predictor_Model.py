import mysql.connector
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Connect to MySQL database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="karthi",
    database="elevator_predictor"
)

# Function to fetch data from MySQL tables
def fetch_data(table_name):
    cursor = db.cursor()
    cursor.execute("SELECT * FROM {}".format(table_name))
    records = cursor.fetchall()
    return records

# Fetch data from MainData and SensorX tables
main_data = fetch_data("MainData")
sensor_data = fetch_data("SensorX")

# Separate features (x1, x2, x3, x4, x5) and labels (revolutions) from the fetched data
X = np.array([[record[1], record[2], record[3], record[4], record[5]] for record in sensor_data])
y = np.array([record[1] for record in main_data])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model with Mean Absolute Error as loss function
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_squared_error'])

# Define early stopping and model checkpoint callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_model.keras", monitor='val_loss', verbose=1, save_best_only=True)

# Train the model with callbacks
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, checkpoint])

# Evaluate the model
loss, mse = model.evaluate(X_test, y_test)
print("Mean Absolute Error:", loss)
print("Mean Squared Error:", mse)
