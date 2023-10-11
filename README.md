# Computer Vision with [MediaPipe](https://developers.google.com/mediapipe)

## Previews
![1](./previews/1.png)

![2](./previews/2.png)

![3](./previews/3.png)
## Building Binary

```pyinstaller --clean --onefile --windowed --name AIVision  --add-data "styles.css;."  --add-data "assets;assets" --icon "./assets/app_icon.png" --collect-data "mediapipe" app.py```

or

```pyinstaller AIVision.spec```