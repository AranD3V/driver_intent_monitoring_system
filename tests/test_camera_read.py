# test_camera_read.py
import cv2
import time

print("Testing camera initialization...\n")

for cam_id in [0, 2]:
    print(f"Testing camera {cam_id}:")
    cap = cv2.VideoCapture(cam_id)

    if not cap.isOpened():
        print(f"  ✗ Cannot open camera {cam_id}")
        continue

    print(f"  ✓ Camera opened")

    # Try reading immediately
    ret, frame = cap.read()
    if ret:
        print(f"  ✓ First read SUCCESS: {frame.shape}")
    else:
        print(f"  ✗ First read FAILED")

        # Wait and retry
        time.sleep(0.5)
        ret, frame = cap.read()
        if ret:
            print(f"  ✓ Second read SUCCESS (needed warmup): {frame.shape}")
        else:
            print(f"  ✗ Second read FAILED")

    cap.release()
    print()
