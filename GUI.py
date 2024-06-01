import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel, QMessageBox
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt, QTimer
import tensorflow as tf
import numpy as np

class InputApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Input Application')
        self.setWindowFlags(Qt.FramelessWindowHint)  # Remove window frame
        self.setGeometry(0, 0, QApplication.desktop().screenGeometry().width(), QApplication.desktop().screenGeometry().height())  # Open in full screen

        self.layout = QVBoxLayout()

        # Add close button
        close_button = QPushButton('✕', self)
        close_button.setFont(QFont('Arial', 24))
        close_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: white;
                border: none;
                padding: 15px;
            }
            QPushButton:hover {
                color: #FF6347;  /* Changed hover color to red */
            }
        """)
        close_button.setGeometry(self.width() - 60, 20, 30, 30)
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

        self.setLayout(self.layout)

        # Start the animation
        self.startAnimation()

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
        model = tf.keras.models.load_model(r"C:/Users/karth/Downloads/Elevator_Predictor/elevator_predictor_model_mae.h5")

        # Create input array for prediction
        new_input_data = np.array([[revolutions, humidity, x1, x2, x3, x4, x5]])

        # Make predictions
        predictions = model.predict(new_input_data)

        vibration_level = predictions[0][0]
        print("Predicted vibration:", vibration_level)

        # Check if vibration level crosses a threshold
        threshold = 0.5  # Example threshold value
        if vibration_level > threshold:
            self.showPopUp(vibration_level)

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

    def showPopUp(self, vibration_level):
        self.pop_up = PopUpWindow(vibration_level)
        self.pop_up.show()

class PopUpWindow(QWidget):
    def __init__(self, vibration_level):
        super().__init__()
        self.setWindowTitle('Warning')
        self.setGeometry(200, 200, 400, 200)

        layout = QVBoxLayout()

        label = QLabel(f"The predicted vibration level is {vibration_level}.")
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont('Arial', 16))
        layout.addWidget(label)

        self.setLayout(layout)

        # Close the pop-up window after 3 seconds
        QTimer.singleShot(8000, self.close)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = InputApp()
    window.showFullScreen()  # Open in full screen
    sys.exit(app.exec_())
