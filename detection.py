from ultralytics import YOLO
from collections import Counter
import cv2

print("Loading YOLO model...")

# Load YOLO model
model = YOLO("yolo11n.pt")

print("Model loaded successfully!")

# Detect objects
results = model("pic.jpg")

# Counter for detected objects
object_counts = Counter()

for result in results:

    # Get every detected object
    for box in result.boxes:

        # Get class ID
        class_id = int(box.cls[0])

        # Convert class ID into object name
        class_name = model.names[class_id]

        # Increase count
        object_counts[class_name] += 1

    # Draw boxes and labels on image
    annotated_image = result.plot()

    # Save result image
    cv2.imwrite("results/detected.jpg", annotated_image)


# Display object counts
print("\n===== DETECTED OBJECTS =====")

for object_name, count in object_counts.items():
    print(f"{object_name}: {count}")

# Total number of objects
total_objects = sum(object_counts.values())

print("----------------------------")
print(f"Total Objects: {total_objects}")
print("============================")

print("\nResult saved in results/detected.jpg")