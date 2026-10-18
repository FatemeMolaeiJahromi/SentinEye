import cv2
import mediapipe as mp
import numpy as np


class ShoulderSurfingDetector:
    def __init__(self, max_faces=4, min_detection_confidence=0.6):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.6,
        )

    def _estimate_head_orientation(self, landmarks, frame_shape):
        h, w = frame_shape[:2]

        model_points = np.array(
            [
                (0.0, 0.0, 0.0),          # nose pick
                (0.0, -330.0, -65.0),     # Chin
                (-225.0, 170.0, -135.0),  # left eye outer corner
                (225.0, 170.0, -135.0),   # right eye outer corner
                (-150.0, -150.0, -125.0), # left side of mouth corner
                (150.0, -150.0, -125.0),  # right side of mouth corner
            ],
            dtype=np.float64,
        )

        idx_list = [1, 152, 263, 33, 287, 57]
        image_points = []
        for idx in idx_list:
            lm = landmarks[idx]
            image_points.append((lm.x * w, lm.y * h))
        image_points = np.array(image_points, dtype=np.float64)

        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vector, _ = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return 0.0, 0.0

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)

        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)

        # rad to degree converter
        yaw_deg = float(np.degrees(yaw))
        pitch_deg = float(np.degrees(pitch))
        return yaw_deg, pitch_deg

    def process(self, frame):
        """
        پردازش فریم ورودی و استخراج کاربر اصلی و تهدیدات سرک‌کشیدن (Shoulder Surfing)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        h, w = frame.shape[:2]

        threat_detected = False
        threat_count = 0
        faces_data = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                xs = [lm.x * w for lm in face_landmarks.landmark]
                ys = [lm.y * h for lm in face_landmarks.landmark]

                x_min = max(0, int(min(xs)))
                y_min = max(0, int(min(ys)))
                x_max = min(w - 1, int(max(xs)))
                y_max = min(h - 1, int(max(ys)))

                bbox = (x_min, y_min, x_max, y_max)
                area = max(0, (x_max - x_min) * (y_max - y_min))

                yaw, pitch = self._estimate_head_orientation(
                    face_landmarks.landmark, frame.shape
                )

                faces_data.append(
                    {
                        "bbox": bbox,
                        "area": area,
                        "yaw": yaw,
                        "pitch": pitch,
                    }
                )

            faces_data.sort(key=lambda x: x["area"], reverse=True)

            for idx, face in enumerate(faces_data):
                if idx == 0:
                    face["is_primary"] = True
                    face["is_threat"] = False
                else:
                    face["is_primary"] = False
                    is_looking_at_screen = abs(face["yaw"]) < 32 and abs(face["pitch"]) < 30
                    face["is_threat"] = is_looking_at_screen
                    if is_looking_at_screen:
                        threat_detected = True
                        threat_count += 1

        return {
            "threat_detected": threat_detected,
            "threat_count": threat_count,
            "faces": faces_data,
        }

    def process_frame(self, frame):
        return self.process(frame)