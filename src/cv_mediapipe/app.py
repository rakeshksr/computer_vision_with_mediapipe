import sys

import cv2
from PySide6.QtCore import QSize, Qt, QThread, Signal, Slot
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cv_mediapipe.ml_applications import MlApplications
from cv_mediapipe.apath import ASSETS_PATH


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
    if callable(getattr(MlApplications, attribute))
    and attribute.startswith("_") is False
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
            # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
            frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            output_image = func(frame_rgb)

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
        self.setWindowTitle("AI Vision")
        # self.setGeometry(0, 0, 800, 500)
        wIcon = QIcon(str(ASSETS_PATH / "app_icon.png"))
        self.setWindowIcon(wIcon)

        # Create a label for the display camera
        self.label = QLabel(self)
        self.label.setFixedSize(640, 480)

        # Thread in charge of updating the image
        self.th = Thread(self)
        self.th.finished.connect(self.close)
        self.th.updateFrame.connect(self.setImage)

        self.combobox = QComboBox(enabled=False)
        self.combobox.addItems(ML_FUNCTIONS.keys())
        qcb_icon = (
            str(ASSETS_PATH).replace("\\", "/") + "/downarrow.png"
        )  # QT CSS only accepting path with "/" seperator
        self.combobox.setStyleSheet(f"""QComboBox::down-arrow {{
            image: url({qcb_icon});
            }}""")

        self.start_button = QPushButton(
            "Start",
            objectName="start",
            icon=QIcon(str(ASSETS_PATH / "play_circle.png")),
            iconSize=QSize(25, 25),
        )
        self.pause_button = QPushButton(
            "Pause",
            objectName="pause",
            icon=QIcon(str(ASSETS_PATH / "pause_circle.png")),
            iconSize=QSize(25, 25),
            visible=False,
        )

        right_layout = QHBoxLayout()
        right_layout.addWidget(self.combobox, stretch=3)
        right_layout.addWidget(self.start_button, stretch=1)
        right_layout.addWidget(self.pause_button, stretch=1)

        # Main layout
        layout = QVBoxLayout()
        layout.addWidget(self.label, alignment=Qt.AlignCenter)
        layout.addLayout(right_layout)

        # Central widget
        widget = QWidget(self)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Interactions
        self.start_button.clicked.connect(self.start_button_handle_click)
        self.pause_button.clicked.connect(self.pause_button_handle_click)
        self.combobox.currentTextChanged.connect(self.set_model)

        # Load the style sheet
        with open(ASSETS_PATH / "styles.css") as f:
            self.setStyleSheet(f.read())

    @Slot()
    def set_model(self, text):
        self.th.set_detect_function(text)

    @Slot()
    def start_button_handle_click(self):
        self.th.set_detect_function(self.combobox.currentText())
        if not self.th.isRunning():
            self.th.start()
        self.start_button.setVisible(False)
        self.pause_button.setVisible(True)
        self.combobox.setEnabled(True)

    @Slot()
    def pause_button_handle_click(self):
        self.th.set_detect_function("No Detection")
        self.start_button.setVisible(True)
        self.pause_button.setVisible(False)
        self.combobox.setEnabled(False)

    @Slot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))
