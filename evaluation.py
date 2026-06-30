import os
import cv2
import pandas as pd

from recognize_face import (
    recognize_face,
    load_database
)

# ----------------------------------
# Paths
# ----------------------------------

TEST_DIR = "data/processed/test"
DATABASE_PATH = "data/embeddings/average_face_database.pkl"

OUTPUT_CSV = "evaluation_results_35_avg.csv"

THRESHOLD = 0.35
METHOD = "Avg Embeddings"

# ----------------------------------
# Load Database
# ----------------------------------

database = load_database(DATABASE_PATH)

# ----------------------------------
# Store Results
# ----------------------------------

results = []

# ----------------------------------
# Evaluate
# ----------------------------------

for label in os.listdir(TEST_DIR):

    label_path = os.path.join(TEST_DIR, label)

    if not os.path.isdir(label_path):
        continue

    for image_name in os.listdir(label_path):

        image_path = os.path.join(label_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        result = recognize_face(
            image,
            database,
            threshold=THRESHOLD
        )

        prediction = result["label"]
        similarity = result["similarity"]

        correct = prediction.lower() == label.lower()

        results.append({
            "image": image_name,
            "ground_truth": label,
            "prediction": prediction,
            "similarity": round(similarity, 4),
            "threshold": THRESHOLD,
            "method": METHOD,
            "correct": correct
        })

        print("=" * 50)
        print(f"Image        : {image_name}")
        print(f"Ground Truth : {label}")
        print(f"Prediction   : {prediction}")
        print(f"Similarity   : {similarity:.4f}")
        print(f"Correct      : {correct}")

# ----------------------------------
# Save Results
# ----------------------------------

df = pd.DataFrame(results)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\n----------------------------------")
print(f"Evaluation completed.")
print(f"Results saved to {OUTPUT_CSV}")
print("----------------------------------")