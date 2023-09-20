import cv2

# Create a VideoCapture object
cap = cv2.VideoCapture(0)  # 0 represents the default webcam

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get the frames per second (fps) of the webcam
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frames per second: {fps}")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the captured frame
    cv2.imshow("Video Capture", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()