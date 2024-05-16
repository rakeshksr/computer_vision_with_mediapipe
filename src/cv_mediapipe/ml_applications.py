import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from cv_mediapipe.apath import ASSETS_PATH
from cv_mediapipe.utils import (
    draw_face_detections_on_image,
    draw_face_landmarks_on_image,
    draw_hand_landmarks_on_image,
    draw_object_detections_on_image,
    draw_pose_landmarks_on_image,
)


class MlApplications:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.drawing_spec = self.mp_drawing.DrawingSpec(
            thickness=1,
            circle_radius=1,
        )
        self.face_detection_model = None
        self.face_mesh_model = None
        self.hand_landmarks_detection_model = None
        self.holistic_detection_model = None
        self.pose_detection_model = None
        self.selfie_segmentation_model = None
        self.object_detetction_model = None

    def no_detection(self, frame_rgb):
        return frame_rgb

    def face_detection(self, frame_rgb):
        if self.face_detection_model is None:
            base_options = python.BaseOptions(model_asset_path= ASSETS_PATH / "face_detector.tflite")
            options = vision.FaceDetectorOptions(
                base_options=base_options,
            )
            self.face_detection_model = vision.FaceDetector.create_from_options(options)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = self.face_detection_model.detect(mp_image)
        image = mp_image.numpy_view()
        annotated_image = draw_face_detections_on_image(image, detection_result)
        return annotated_image

    def face_mesh_detection(self, frame_rgb):
        if self.face_mesh_model is None:
            base_options = python.BaseOptions(
                model_asset_path= ASSETS_PATH / "face_landmarker_with_blendshapes.task"
            )
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=1,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
            self.face_mesh_model = vision.FaceLandmarker.create_from_options(options)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = self.face_mesh_model.detect(mp_image)
        image = mp_image.numpy_view()
        annotated_image = draw_face_landmarks_on_image(image, detection_result)
        return annotated_image

    def hand_landmarks_detection(self, frame_rgb):
        if self.hand_landmarks_detection_model is None:
            base_options = python.BaseOptions(model_asset_path= ASSETS_PATH / "hand_landmarker.task")
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
            )
            self.hand_landmarks_detection_model = (
                vision.HandLandmarker.create_from_options(options)
            )
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = self.hand_landmarks_detection_model.detect(mp_image)
        image = mp_image.numpy_view()
        annotated_image = draw_hand_landmarks_on_image(image, detection_result)
        return annotated_image

    # To-Do https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker
    # def holistic_detection(self, frame_rgb):
    #     pass

    def pose_detection(self, frame_rgb):
        if self.pose_detection_model is None:
            base_options = python.BaseOptions(
                model_asset_path= ASSETS_PATH / "pose_landmarker.task"
            )
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=True,
            )
            self.pose_detection_model = vision.PoseLandmarker.create_from_options(
                options
            )
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = self.pose_detection_model.detect(mp_image)
        image = mp_image.numpy_view()
        annotated_image = draw_pose_landmarks_on_image(image, detection_result)
        return annotated_image

    def selfie_segmention(self, frame_rgb):
        if self.selfie_segmentation_model is None:
            base_options = python.BaseOptions(model_asset_path= ASSETS_PATH / "deeplab_v3.tflite")
            options = vision.ImageSegmenterOptions(
                base_options=base_options,
                output_category_mask=True,
            )
            self.selfie_segmentation_model = vision.ImageSegmenter.create_from_options(
                options
            )
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        segmentation_result = self.selfie_segmentation_model.segment(mp_image)
        category_mask = segmentation_result.category_mask
        condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.1
        bg_image = cv2.blur(frame_rgb, (20, 20))
        annotated_image = np.where(condition, frame_rgb, bg_image)
        return annotated_image

    def object_detection(self, frame_rgb):
        if self.object_detetction_model is None:
            base_options = python.BaseOptions(model_asset_path= ASSETS_PATH / "efficientdet.tflite")
            options = vision.ObjectDetectorOptions(
                base_options=base_options,
                score_threshold=0.5,
            )
            self.object_detetction_model = vision.ObjectDetector.create_from_options(
                options
            )
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = self.object_detetction_model.detect(mp_image)
        image = mp_image.numpy_view()
        annotated_image = draw_object_detections_on_image(image, detection_result)
        return annotated_image
