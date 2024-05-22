# Computer Vision with [MediaPipe](https://ai.google.dev/edge/mediapipe)

## Previews
![1](./previews/1.png)

![2](./previews/2.png)

![3](./previews/3.png)

## Building Steps

1. Install [Rye](https://rye-up.com/guide/installation/)
2. Sync project: `rye sync`
3. Activate environment
    * On Windows, run: `.venv\Scripts\activate/`
    * On Unix or MacOS, run: `source .venv/bin/activate`
4.
    1. To build installer: `rye run build-installer`
    2. To build binary: `rye run build-binary`

> [!NOTE]
> [cargo-packager](https://github.com/crabnebula-dev/cargo-packager) is required to build installer