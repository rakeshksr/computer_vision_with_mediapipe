# Computer Vision with [MediaPipe](https://ai.google.dev/edge/mediapipe)

## Previews
![1](./previews/1.png)

![2](./previews/2.png)

![3](./previews/3.png)

## Building Steps

1. Install [Rye](https://rye-up.com/guide/installation/)
2. Sync project: `rye sync`
3. ```pyinstaller --clean --onefile --windowed --name AIVision  --add-data "src/cv_mediapipe/assets;assets" --icon "src/cv_mediapipe/assets/app_icon.png" src/cv_mediapipe/__main__.py```

    or

    ```pyinstaller AIVision.spec```