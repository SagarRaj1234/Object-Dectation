
from ultralytics import YOLO
import cv2
import cvzone
import math
import json

# cap = cv2.VideoCapture(0)  # for webcam
cap = cv2.VideoCapture("Videos/dog.mp4")  # For Video or images
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("../Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

def generate_json_output(objects):
    """
    Convert object and sub-object detections into a JSON formatted string.

    objects: List of dictionaries with object detection details.
    """
    output_json = []

    for obj in objects:
        object_data = {
            'object': obj['object'],
            'id': obj['id'],
            'bbox': obj['bbox'],
            'subobject': obj['subobject']
        }

        output_json.append(object_data)

    return json.dumps(output_json, indent=4)


while True:
    success, img = cap.read()  # it passes one frame of video
    if not success:
        break  # Exit if video ends or webcam fails

    results = model(img, stream=True)

    detected_objects = []  # List to store detected objects for JSON output

    for r in results:
        boxes1 = r.boxes
        for box in boxes1:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))  # Using cvzone for rectangle

            # Confidence value (rounded to 2 decimal places)
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Get class name and print the detected class info for debugging
            cls1 = int(box.cls[0])

            # Debugging: Print class index, class name, and confidence to ensure it's correct
            print(f"Detected class index: {cls1}, Class Name: {classNames[cls1]}, Confidence: {conf}")

            # Only use the detection if the confidence is above a threshold (0.5)
            if conf > 0.5:  # Change this threshold as needed
                class_name = classNames[cls1]
                cvzone.putTextRect(img, f'{class_name} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

                # Prepare object data for JSON
                object_data = {
                    'object': class_name,
                    'id': len(detected_objects) + 1,  # Increment ID for each object
                    'bbox': [x1, y1, x2, y2],
                    'subobject': {}  # You can add sub-object detection here if required
                }
                detected_objects.append(object_data)

    # Generate JSON output for the current frame
    json_output = generate_json_output(detected_objects)

    # Save JSON output to a file after every frame (overwrite previous content)
    with open('output.json', 'w') as json_file:
        json.dump(json_output, json_file, indent=4)

    # Display the image
    cv2.imshow("Image", img)

    # Wait for a key press (1  for real-time video) or (0 for image)
    cv2.waitKey(1)

# Release the video capture when done
cap.release()
cv2.destroyAllWindows()






