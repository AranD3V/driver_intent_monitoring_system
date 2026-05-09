# identify_cameras.py
import cv2
import time

print("Scanning all cameras...\n")

cameras = {}

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(i)

    if cap.isOpened():
        # Give it time to initialize
        time.sleep(1)
        ret, frame = cap.read()

        if ret and frame is not None:
            h, w = frame.shape[:2]
            cameras[i] = {'width': w, 'height': h, 'works': True}
            print(f"✓ Camera {i}: {w}×{h}")
        else:
            cameras[i] = {'works': False}
            print(f"⚠ Camera {i}: Opens but can't read")

        cap.release()
    else:
        print(f"✗ Camera {i}: Not available")

print("\n" + "="*50)
print("CAMERA IDENTIFICATION")
print("="*50)

working = [idx for idx, info in cameras.items() if info.get('works')]

if len(working) >= 2:
    print(f"\nYou have {len(working)} working cameras: {working}")
    print("\nRECOMMENDED SETUP:")
    print(f"  --driver {working[0]}  (for driver face)")
    print(f"  --scene {working[1]}   (for road/scene)")

    if len(working) > 2:
        print(f"\n  Camera {working[2]} is available but not needed")
        print(f"  (Moto Smart Connect can be disconnected)")
else:
    print(f"\n⚠ Only {len(working)} camera(s) working")
    print("Need at least 2 cameras for the system")
