import sys
import time
import traceback
import cv2

from core.shoulder_surf import ShoulderSurfingDetector


def main():
    print("1. Opening Camera...", flush=True)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not access the webcam.", flush=True)
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("2. Initializing AI Detector...", flush=True)
    try:
        detector = ShoulderSurfingDetector()
    except Exception:
        print("CRITICAL: Failed to load detector!", flush=True)
        traceback.print_exc()
        cap.release()
        return 1

    print("3. SentinEye is Running! Press 'q' to exit.", flush=True)

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            result = detector.process(frame)
            annotated_frame = frame.copy()

            faces = result.get("faces", [])
            threats = result.get("threat_count", 0)

            for face in faces:
                x1, y1, x2, y2 = face["bbox"]

                if face["is_primary"]:
                    color = (0, 255, 0)     #grren: primary preson
                    label = "Primary User"
                elif face["is_threat"]:
                    color = (0, 0, 255)      # red: alert
                    label = f"THREAT! Yaw: {face['yaw']:.1f}"
                else:
                    color = (0, 255, 255)    # yellow: passanger
                    label = "Bystander"

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(22, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

            now = time.time()
            dt = now - prev_time
            if dt > 0:
                fps = 1.0 / dt
            prev_time = now

            # Security Status
            status_color = (0, 0, 255) if threats > 0 else (0, 255, 0)
            status_text = (
                f"ALERT: {threats} Snooper(s) Looking!"
                if threats > 0
                else "Status: SECURE"
            )

            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f} | Faces: {len(faces)}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                annotated_frame,
                status_text,
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                status_color,
                2,
            )

            cv2.imshow("SentinEye - Privacy Shield", annotated_frame)

            # press "q" to exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Exit requested by user (Key 'q').", flush=True)
                break

    except KeyboardInterrupt:
        print("\nInterrupted by Ctrl+C.", flush=True)
    except Exception:
        print("Runtime exception in loop:", flush=True)
        traceback.print_exc()
        return 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. SentinEye terminated safely.", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
