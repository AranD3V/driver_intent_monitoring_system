# test_cameras.py
import cv2

print("Testing camera indices...\n")

for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            print(f"✓ Camera {i}: WORKS ({w}×{h}) - showing preview, press any key to close")
            cv2.putText(frame, f"Camera {i}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(0)
            cv2.destroyWindow(f"Camera {i}")
        else:
            print(f"⚠ Camera {i}: Opens but can't read frames")
        cap.release()
    else:
        print(f"✗ Camera {i}: Not available")

    if i == 5:
        print()  # blank line for readability
