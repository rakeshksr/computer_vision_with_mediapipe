import sys

from PySide6.QtWidgets import QApplication

from cv_mediapipe.app import MainWindow


def main():
    app: QApplication = QApplication(sys.argv)
    window: MainWindow = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
