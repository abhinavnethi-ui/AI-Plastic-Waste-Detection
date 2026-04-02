import cv2

# Test phone webcam connection
phone_url = "http://192.168.33.247:8080/video"
print(f"Testing connection to: {phone_url}")

cap = cv2.VideoCapture(phone_url)

if cap.isOpened():
    print("✅ Successfully connected to phone webcam!")
    ret, frame = cap.read()
    if ret:
        print("✅ Successfully grabbed a frame!")
        print(f"Frame shape: {frame.shape}")
    else:
        print("❌ Connected but failed to grab frame")
else:
    print("❌ Failed to connect to phone webcam")
    print("Make sure:")
    print("1. IP Webcam app is running on your phone")
    print("2. Phone and computer are on same WiFi")
    print("3. IP address is correct")

cap.release()
cv2.destroyAllWindows()
