import cv2
from ultralytics import YOLO

from recognize_face import (
    recognize_face,
    load_database
)

# ----------------------------------
# Load YOLO Face Detector
# ----------------------------------

model = YOLO("models/yolo/yolov8n-face.pt")

database = load_database(
    "data/embeddings/face_database.pkl"
)

# ----------------------------------
# Display Labels
# ----------------------------------

LABEL_MAP = {
    "kohli": "Virat Kohli",
    "rohit": "Rohit Sharma",
    "msd": "MS Dhoni",
    "bumrah": "Jasprit Bumrah",
    "unknown": "Unknown",
    "No Face": "No Face"
}

# ----------------------------------
# Prediction Function
# ----------------------------------

def predict_image(image):

    if image is None:
        raise ValueError("Could not read image")

    predictions = []

    # Detect Faces
    results = model(image)[0]

    # Process Every Face
    for i, box in enumerate(results.boxes, start=1):

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        face = image[y1:y2, x1:x2]

        if face.size == 0:
            continue

        # result = recognize_face(face)
        result = recognize_face(face,database,threshold=0.35)
        
        label = result["label"]
        similarity = result["similarity"]
        display_label = LABEL_MAP.get(label, label)
        predictions.append({
            "face_id": i,
            "label": label,
            "similarity": float(similarity)
        })

        print(f"\nFace {i}")
        print(f"Label      : {label}")
        print(f"Similarity : {similarity:.4f}")
        print("-" * 30)

        if label == "unknown":
            color = (0, 0, 255)      # Red (BGR)
        else:
            color = (0, 255, 0)      # Green (BGR)

        # Draw Bounding Box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # Draw Label
        cv2.putText(
            image,
            f"{display_label}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return image, predictions


# ----------------------------------
# Test
# ----------------------------------

if __name__ == "__main__":

    image = cv2.imread("data/test/ms_yuvi_test.jpeg")

    output, predictions = predict_image(image)

    cv2.imwrite("prediction.jpg", output)

    print("\nPrediction Summary")
    print("=" * 40)

    for prediction in predictions:
        print(
            f"Face {prediction['face_id']} | "
            f"Label: {prediction['label']} | "
            f"Similarity: {prediction['similarity']:.4f}"
        )

    print("\nPrediction saved as prediction.jpg")