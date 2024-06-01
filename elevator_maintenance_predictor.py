import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel, QMessageBox
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt, QTimer
import numpy as np
import mysql.connector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

class InputApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Input Application')
        self.setWindowFlags(Qt.FramelessWindowHint)  # Remove window frame

        self.layout = QVBoxLayout()

        # Add close button
        close_button = QPushButton('✕', self)
        close_button.setFont(QFont('Arial', 24))
        close_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: white;
                border: none;
                padding: 3px;
            }
            QPushButton:hover {
                color: #FF6347;  /* Changed hover color to red */
            }
        """)
        close_button.clicked.connect(self.closeApp)
        self.layout.addWidget(close_button)

        # Create labels and input fields
        labels = ['Revolutions:', 'Humidity:', 'x1:', 'x2:', 'x3:', 'x4:', 'x5:']  # Removed 'Vibration:'
        self.inputs = []
        for label_text in labels:
            label = QLabel(label_text)
            label.setFont(QFont('Arial', 20, QFont.Bold))  # Increased font size and bold
            label.setStyleSheet("color: #FFFFFF;")  # Changed label text color to white
            self.layout.addWidget(label)

            line_edit = QLineEdit()
            line_edit.setFont(QFont('Arial', 18))
            line_edit.setStyleSheet("""
                QLineEdit {
                    background-color: rgba(255, 255, 255, 200);
                    border: 2px solid transparent;
                    border-radius: 15px;  /* Increased border radius for a smoother look */
                    padding: 15px;  /* Increased padding for input fields */
                    color: #333333;  /* Changed text color to dark gray */
                }
                QLineEdit:focus {
                    border-color: #4CAF50;  /* Change border color when input field is focused */
                    background-color: rgba(255, 255, 255, 230); /* Lighter background when focused */
                }
            """)
            self.layout.addWidget(line_edit)
            self.inputs.append(line_edit)

        # Add submit button
        submit_button = QPushButton('Submit')
        submit_button.setFont(QFont('Arial', 20))
        submit_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 20px;
                padding: 20px 40px;  /* Increased padding for submit button */
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        submit_button.clicked.connect(self.submitClicked)
        self.layout.addWidget(submit_button)

        # Add retrain button
        retrain_button = QPushButton('Retrain Model')
        retrain_button.setFont(QFont('Arial', 20))
        retrain_button.setStyleSheet("""
            QPushButton {
                background-color: #337ab7;
                color: white;
                border-radius: 20px;
                padding: 20px 40px;  /* Increased padding for submit button */
            }
            QPushButton:hover {
                background-color: #286090;
            }
        """)
        retrain_button.clicked.connect(self.retrainModel)
        self.layout.addWidget(retrain_button)

        self.setLayout(self.layout)

        # Start the animation
        self.startAnimation()

        # Connect to the database
        self.connectToDatabase()

    def closeApp(self):
        self.close()

    def submitClicked(self):
        revolutions = float(self.inputs[0].text())
        humidity = float(self.inputs[1].text())
        x1 = float(self.inputs[2].text())
        x2 = float(self.inputs[3].text())
        x3 = float(self.inputs[4].text())
        x4 = float(self.inputs[5].text())
        x5 = float(self.inputs[6].text())

        # Load the trained model
        model = joblib.load(r"C:/Users/karth/Downloads/Elevator_Predictor/elevator_predictor_model_linear_regression.pkl")

        # Create input array for prediction
        new_input_data = np.array([[revolutions, humidity, x1, x2, x3, x4, x5]])

        # Make predictions
        predictions = model.predict(new_input_data)

        vibration_level = predictions[0]
        print("Predicted vibration:", vibration_level)

        # Insert new inputs and predictions into the database
        self.insertIntoDatabase(revolutions, humidity, x1, x2, x3, x4, x5, vibration_level)

        # Check if vibration level crosses a threshold
        threshold = 20
        if vibration_level > threshold:
            self.showPopUp("Elevator needs to be serviced")
        else:
            # Clear all input fields
            for line_edit in self.inputs:
                line_edit.clear()

    def startAnimation(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animateBackground)
        self.timer.start(1000)  # Set the timer interval in milliseconds

    def animateBackground(self):
        # Transition from black to dark blue (#00008B)
        start_color = QColor("#000000")
        end_color = QColor("#00008B")

        # Calculate the current interpolated color
        factor = int(100 - (self.timer.remainingTime() / 10))
        interpolated_color = start_color.lighter(factor if factor >= 0 else 0).name()

        # Change background color gradually
        self.setStyleSheet(f"""
            InputApp {{
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 {interpolated_color}, stop: 1 {end_color.name()});
            }}
        """)

    def showPopUp(self, message):
        self.pop_up = PopUpWindow(message)
        self.pop_up.show()

    def connectToDatabase(self):
        # Establish a connection to the database
        self.db_connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="karthi",
            database="elevator_predictor"
        )

    def insertIntoDatabase(self, revolutions, humidity, x1, x2, x3, x4, x5, vibration_level):
        # Convert numpy.float64 values to Python floats
        revolutions = float(revolutions)
        humidity = float(humidity)
        x1 = float(x1)
        x2 = float(x2)
        x3 = float(x3)
        x4 = float(x4)
        x5 = float(x5)
        vibration_level = float(vibration_level)

        # Create a cursor object to execute SQL queries
        cursor = self.db_connection.cursor()

        # Insert new inputs and predictions into the appropriate tables
        insert_query = "INSERT INTO MainData (revolutions, humidity, vibration) VALUES (%s, %s, %s)"
        cursor.execute(insert_query, (revolutions, humidity, vibration_level))

        insert_query = "INSERT INTO SensorX (x1, x2, x3, x4, x5) VALUES (%s, %s, %s, %s, %s)"
        cursor.execute(insert_query, (x1, x2, x3, x4, x5))

        # Commit changes to the database
        self.db_connection.commit()

        # Close the cursor
        cursor.close()

    def closeEvent(self, event):
        # Close the database connection when the application is closed
        self.db_connection.close()

    def retrainModel(self):
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

        # Prepare data
        features = []
        labels = []
        for row in data:
            features.append(list(row[:-1]))
            labels.append(row[-1])

        # Convert lists to numpy arrays
        features = np.array(features)
        labels = np.array(labels)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)

        # Create and train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        predictions = model.predict(X_test)

        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_test, predictions)
        print("Mean Absolute Error:", mae)

        # Save the model
        joblib.dump(model, "elevator_predictor_model_linear_regression.pkl")

        # Inform the user that the model has been retrained
        QMessageBox.information(self, "Model Retrained", f"The model has been retrained with Mean Absolute Error: {mae}")

class PopUpWindow(QWidget):
    def __init__(self, message):
        super().__init__()
        self.setWindowTitle('Warning')
        self.setGeometry(200, 200, 400, 200)

        layout = QVBoxLayout()

        label = QLabel(message)
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont('Arial', 16))
        layout.addWidget(label)

        self.setLayout(layout)

        # Close the pop-up window after 3 seconds
        QTimer.singleShot(10000, self.close)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = InputApp()
    # Get the screen size
    screen_geometry = app.primaryScreen().geometry()
    window_width = min(screen_geometry.width(), 800)
    window_height = int(window_width * 3 / 4)  # Aspect ratio of 4:3
    window.setGeometry(0, 0, window_width, window_height)
    window.showFullScreen()  # Open in full screen
    sys.exit(app.exec_())
