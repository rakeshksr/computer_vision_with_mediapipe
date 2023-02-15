import numpy as np
import cv2
import mediapipe as mp


class MlApplications():

    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.drawing_spec = self.mp_drawing.DrawingSpec(
            thickness=1,
            circle_radius=1,
        )

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection_model = None

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh_model = None

        self.mp_hands = mp.solutions.hands
        self.hand_landmarks_detection_model = None

        self.mp_holistic = mp.solutions.holistic
        self.holistic_detection_model = None

        self.mp_objectron = mp.solutions.objectron
        self.objectron_detection_models = None
        self.objectron_classes = {"Camera", "Chair", "Cup", "Shoe"}

        self.mp_pose_detection = mp.solutions.pose
        self.pose_detection_model = None

        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation_model = None

    def no_detection(self, frame):
        return cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

    def face_detection(self, frame):
        if self.face_detection_model is None:
            self.face_detection_model = self.mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5,
            )
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame_rgb.flags.writeable = False
        results = self.face_detection_model.process(frame_rgb)

        # Draw the face detection annotations on the image.
        frame_rgb.flags.writeable = True
        if results.detections:
            for detection in results.detections:
                self.mp_drawing.draw_detection(frame_rgb, detection)
        return frame_rgb

    def face_mesh_detection(self, frame):
        if self.face_mesh_model is None:
            self.face_mesh_model = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.face_mesh_model.process(frame_rgb)
        frame_rgb.flags.writeable = True
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=frame_rgb,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style(),
                )
                self.mp_drawing.draw_landmarks(
                    image=frame_rgb,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )
                self.mp_drawing.draw_landmarks(
                    image=frame_rgb,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                )
        return frame_rgb

    def hand_landmarks_detection(self, frame):
        if self.hand_landmarks_detection_model is None:
            self.hand_landmarks_detection_model = self.mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hand_landmarks_detection_model.process(frame_rgb)
        frame_rgb.flags.writeable = True
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=frame_rgb,
                    landmark_list=hand_landmarks,
                    connections=self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style(),
                )
        return frame_rgb

    def holistic_detection(self, frame):
        if self.holistic_detection_model is None:
            self.holistic_detection_model = self.mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.holistic_detection_model.process(frame_rgb)
        frame_rgb.flags.writeable = True
        # if results.segmentation_mask:
        #     condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        #     bg_image = np.zeros(image.shape, dtype=np.uint8)
        #     bg_image[:] = [192,192,192]
        #     image = np.where(condition, image, bg_image)
        self.mp_drawing.draw_landmarks(
            image=frame_rgb,
            landmark_list=results.face_landmarks,
            connections=self.mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=self.drawing_spec,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style(),
        )
        self.mp_drawing.draw_landmarks(
            image=frame_rgb,
            landmark_list=results.face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=self.drawing_spec,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        self.mp_drawing.draw_landmarks(
            image=frame_rgb,
            landmark_list=results.left_hand_landmarks,
            connections=self.mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style())
        self.mp_drawing.draw_landmarks(
            image=frame_rgb,
            landmark_list=results.right_hand_landmarks,
            connections=self.mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style())
        self.mp_drawing.draw_landmarks(
            image=frame_rgb,
            landmark_list=results.pose_landmarks,
            connections=self.mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
        )
        # self.mp_drawing.draw_landmarks(
        #     image=frame_rgb,
        #     landmark_list=results.pose_world_landmarks,
        #     connections=self.mp_holistic.POSE_CONNECTIONS,
        #     landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
        # )
        return frame_rgb

    def objectron_detection(self, frame):
        if self.objectron_detection_models is None:
            self.objectron_detection_models = [
                self.mp_objectron.Objectron(
                    static_image_mode=True,
                    max_num_objects=5,
                    min_detection_confidence=0.5,
                    model_name=objectron_class,
                )
                for objectron_class in self.objectron_classes
            ]
        frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = [objectron_model.process(frame_rgb) for objectron_model in self.objectron_detection_models]
        frame_rgb.flags.writeable = True
        for result in results:
            if result.detected_objects:
                for detected_object in result.detected_objects:
                    self.mp_drawing.draw_landmarks(
                        frame_rgb,
                        detected_object.landmarks_2d,
                        self.mp_objectron.BOX_CONNECTIONS,
                    )
                    self.mp_drawing.draw_axis(
                        frame_rgb,
                        detected_object.rotation,
                        detected_object.translation,
                    )
        return frame_rgb

    def pose_detection(self, frame):
        if self.pose_detection_model is None:
            self.pose_detection_model = self.mp_pose_detection.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.pose_detection_model.process(frame_rgb)
        frame_rgb.flags.writeable = True
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image=frame_rgb,
                landmark_list=results.pose_landmarks,
                connections=self.mp_pose_detection.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
            )
        # self.mp_drawing.draw_landmarks(
        #     image=frame_rgb,
        #     landmark_list=results.pose_world_landmarks,
        #     connections=self.mp_holistic.POSE_CONNECTIONS,
        #     landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
        # )
        return frame_rgb

    def selfie_segmention(self, frame):
        if self.selfie_segmentation_model is None:
            self.selfie_segmentation_model = self.mp_selfie_segmentation.SelfieSegmentation(
                model_selection=1,
            )
        frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.selfie_segmentation_model.process(frame_rgb)
        frame_rgb.flags.writeable = True
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = cv2.blur(frame_rgb, (20,20))
        frame_rgb = np.where(condition, frame_rgb, bg_image)
        return frame_rgb
