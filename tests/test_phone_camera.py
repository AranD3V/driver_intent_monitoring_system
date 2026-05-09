# test_phone_camera.py
import cv2
import time

print("Testing Moto Smart Connect camera...\n")

# Try different backends
backends = [
    (cv2.CAP_DSHOW, "DSHOW"),
    (cv2.CAP_ANY, "ANY/Default"),
]

for backend_id, backend_name in backends:
    print(f"Trying {backend_name} backend:")
    cap = cv2.VideoCapture(1, backend_id)

    if not cap.isOpened():
        print(f"  ✗ Cannot open with {backend_name}")
        continue

    print(f"  ✓ Camera opened")

    # Phone cameras need longer warmup
    print(f"  ⏳ Waiting for stream (15 seconds)...")
    time.sleep(15)  # Give phone app time to start streaming

    # Try reading frames
    success_count = 0
    for i in range(30):
        ret, frame = cap.read()
        if ret and frame is not None:
            success_count += 1
            if i == 0:
                print(f"  ✓ First frame received: {frame.shape}")
        time.sleep(0.1)

    print(f"  → Success rate: {success_count}/30 frames")

    if success_count > 20:
        print(f"  ✓ {backend_name} works!")
        print(f"\n  Showing live preview...")

        for _ in range(100):
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f'Moto Smart Connect - {backend_name}', frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()
        break
    else:
        print(f"  ✗ {backend_name} unreliable\n")

    cap.release()

print("\nTest complete")
