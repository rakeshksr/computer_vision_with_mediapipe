import sys

import cv2

from PySide6.QtCore import Qt, QSize, QThread, Signal, Slot
from PySide6.QtGui import QAction, QColor, QIcon, QImage, QPalette, QKeySequence, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QGroupBox,
                               QHBoxLayout, QLabel, QMainWindow, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from ml_applications import MlApplications

def convert_snake_case_to_title_case(snake_case_str: str) -> str:
    """
    Converts a string from snake_case to Title Case.
    """
    # Split the string into individual words
    words = snake_case_str.split("_")

    # Capitalize the first letter of each word and join them with a space
    title_case_str = " ".join(word.capitalize() for word in words)

    return title_case_str

ML_FUNCTIONS = {
    convert_snake_case_to_title_case(attribute): attribute
    for attribute in dir(MlApplications)
    if callable(getattr(MlApplications, attribute)) and attribute.startswith('__') is False
}

class Thread(QThread):
    updateFrame = Signal(QImage)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.detect_function = None
        self.status = True
        self.cap = True
        self.mla = MlApplications()

    def set_detect_function(self, fname):
        self.detect_function = ML_FUNCTIONS[fname]

    def run(self):
        self.cap = cv2.VideoCapture(0)
        while self.status:
            ret, frame = self.cap.read()
            if not ret:
                continue
            func = getattr(self.mla, self.detect_function)
            output_image = func(frame)
            # Creating and scaling QImage
            h, w, ch = output_image.shape
            img = QImage(output_image.data, w, h, ch * w, QImage.Format_RGB888)
            scaled_img = img.scaled(640, 480, Qt.KeepAspectRatio)

            # Emit signal
            self.updateFrame.emit(scaled_img)
        sys.exit(-1)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Title and dimensions
        self.setWindowTitle("Compute Vision using Deep Learning")
        # self.setGeometry(0, 0, 800, 500)

        # # Main menu bar
        # self.menu = self.menuBar()
        # self.menu_file = self.menu.addMenu("File")
        # exit = QAction("Exit", self, triggered=qApp.quit)
        # self.menu_file.addAction(exit)

        # self.menu_about = self.menu.addMenu("&About")
        # about = QAction("About Qt", self, shortcut=QKeySequence(QKeySequence.HelpContents),
        #                 triggered=qApp.aboutQt)
        # self.menu_about.addAction(about)

        # Create a label for the display camera
        self.label = QLabel(self)
        self.label.setFixedSize(640, 480)
 

        # Thread in charge of updating the image
        self.th = Thread(self)
        self.th.finished.connect(self.close)
        self.th.updateFrame.connect(self.setImage)

        # Model group
        # self.group_model = QGroupBox("Trained model")
        # self.group_model.setSizePolicy(
        #     QSizePolicy.Preferred, QSizePolicy.Expanding)
        # model_layout = QHBoxLayout()

        self.combobox = QComboBox()
        self.combobox.addItems(ML_FUNCTIONS.keys())
        # for fun in ML_FUNCTIONS:
        #     self.combobox.addItem(fun)
        # for xml_file in os.listdir(cv2.data.haarcascades):
        #     if xml_file.endswith(".xml"):
        #         self.combobox.addItem(xml_file)

        # model_layout.addWidget(QLabel("File:"), 10)
        # model_layout.addWidget(self.combobox, 90)
        # self.group_model.setLayout(model_layout)

        # Buttons layout
        # buttons_layout = QHBoxLayout()
        # self.button1 = QPushButton("Start")
        # self.button2 = QPushButton("Stop")
        # self.button1.setSizePolicy(
        #     QSizePolicy.Preferred, QSizePolicy.Expanding)
        # self.button2.setSizePolicy(
        #     QSizePolicy.Preferred, QSizePolicy.Expanding)
        # buttons_layout.addWidget(self.button2)
        # buttons_layout.addWidget(self.button1)
        self.start_pause_button = QPushButton()
        self.handle_click()
        # self.handle_click(self.start_pause_button)
        # buttons_layout.addWidget(self.start_stop_button)


        right_layout = QHBoxLayout()
        # right_layout.addWidget(self.group_model, 1)
        # right_layout.addLayout(buttons_layout, 1)
        right_layout.addWidget(self.combobox, stretch=3)
        right_layout.addWidget(self.start_pause_button, stretch=1)

        # Main layout
        layout = QVBoxLayout()
        layout.addWidget(self.label, alignment=Qt.AlignCenter)
        layout.addLayout(right_layout)

        # Central widget
        widget = QWidget(self)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Connections
        # self.button1.clicked.connect(self.start)
        # self.button2.clicked.connect(self.pause_detection)
        self.start_pause_button.clicked.connect(self.handle_click)
        self.combobox.currentTextChanged.connect(self.set_model)

        # Load the style sheet
        with open("styles.css") as f:
            self.setStyleSheet(f.read())

    def set_button_properties(self, button):
        pass


    @Slot()
    def set_model(self, text):
        self.th.set_detect_function(text)

    @Slot()
    def handle_click(self):
        if self.start_pause_button.text() == "Start":
            self.start_detection()
            text = "Pause"
            bg_color = "#bb85f1" 
            icon = "./assets/pause_circle.png"
            cb = True
        else:
            if self.start_pause_button.text() != "":
                self.stop_detection()
            text = "Start"
            bg_color = "#47c2a7"
            icon = "./assets/play_circle.png"
            cb = False
        self.start_pause_button.setText(text)
        self.start_pause_button.setStyleSheet(f"background-color: {bg_color};")
        self.start_pause_button.setIcon(QIcon(icon))
        self.start_pause_button.setIconSize(QSize(25,25))
        self.combobox.setEnabled(cb)

    @Slot()
    def stop_detection(self):
        print("Finishing...")
        self.th.set_detect_function("No Detection")

    @Slot()
    def start_detection(self):
        print("Starting...")
        self.th.set_detect_function(self.combobox.currentText())
        if not self.th.isRunning():
            self.th.start()

    @Slot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
