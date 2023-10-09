# Computer Vision with [MediaPipe](https://developers.google.com/mediapipe)

## PyInstaller

### Create single exutable

```pyinstaller --clean --onefile --windowed --name AIVision  --add-data "styles.css;."  --add-data "assets;assets" --icon "./assets/app_icon.png" --collect-data "mediapipe" app.py```

or

```pyinstaller AIVision.spec```