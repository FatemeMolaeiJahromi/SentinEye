import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open web camera.")
else:
    print("Success! Camera is open.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame.")
            break
        cv2.imshow('Test Cam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
