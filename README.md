# Computer Vision with Deep Learning

## PyInstaller

### Create single exutable

```pyinstaller --clean --onefile --windowed --name AI  --add-data "styles.css;."  --add-data "assets;assets" --icon "./assets/app_icon.png" --collect-data "mediapipe" app.py```

or

```pyinstaller.exe AI.spec```