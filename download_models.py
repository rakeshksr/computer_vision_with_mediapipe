from dataclasses import dataclass
from pathlib import Path
import requests

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

DOWNLOAD_PATH = Path("src") / "cv_mediapipe" / "assets"

@dataclass
class ModelProps:
    name: str
    reference_url: str
    download_url: str

MODELS_DATA = [
    ModelProps(
        name="face_detector.tflite",
        reference_url = "https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector#models",
        download_url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
    ),
        ModelProps(
        name="face_landmarker_with_blendshapes.task",
        reference_url = "https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/index#models",
        download_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        ),
        ModelProps(
            name = "hand_landmarker.task",
            reference_url = "https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models",
            download_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        ),
        # ModelProps(
        #     name = "holostic",
        #     reference_url = "",
        #     download_url = ""
        # ),
        ModelProps(
            name = "pose_landmarker.task",
            reference_url = "https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/index#models",
            download_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
            # "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
            # "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
        ),
        ModelProps(
            name="deeplab_v3.tflite",
            reference_url = "https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter#models",
            download_url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/latest/deeplab_v3.tflite"
            # "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
            # "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite"
            # "https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite"
            # "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
        ),
        ModelProps(
            name = "efficientdet.tflite",
            reference_url = "https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector#models",
            download_url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite"
            # "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite"
            # "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite"
            # "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/int8/latest/efficientdet_lite2.tflite"
            # "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float16/latest/efficientdet_lite2.tflite"
            # "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float32/latest/efficientdet_lite2.tflite"
            # "https://storage.googleapis.com/mediapipe-models/object_detector/ssd_mobilenet_v2/float16/latest/ssd_mobilenet_v2.tflite"
            # "https://storage.googleapis.com/mediapipe-models/object_detector/ssd_mobilenet_v2/float32/latest/ssd_mobilenet_v2.tflite"
        )
]

def download_models():
    for model in MODELS_DATA:
        model_path = DOWNLOAD_PATH/model.name
        if not model_path.exists():
            response = requests.get(model.download_url)
            if response.ok:
                with open(model_path, mode="wb") as fio:
                    fio.write(response.content)


class CustomHook(BuildHookInterface):
    def initialize(self, version, build_data):
        download_models()