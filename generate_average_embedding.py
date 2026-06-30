import os
import cv2
import pickle
import numpy as np

from insightface.app import FaceAnalysis

# ----------------------------------
# Initialize ArcFace
# ----------------------------------

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=0)

# ----------------------------------
# Paths
# ----------------------------------

TRAIN_DIR = "data/processed/train"
OUTPUT_PATH = "data/embeddings/average_face_database.pkl"

# ----------------------------------
# Database
# ----------------------------------

database = []

# ----------------------------------
# Generate Average Embeddings
# ----------------------------------

for label in os.listdir(TRAIN_DIR):

    label_path = os.path.join(TRAIN_DIR, label)

    if not os.path.isdir(label_path):
        continue

    print(f"\nProcessing {label}...")

    embeddings = []

    for image_name in os.listdir(label_path):

        image_path = os.path.join(label_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        faces = app.get(image)

        if len(faces) != 1:
            continue

        embeddings.append(faces[0].embedding)

    if len(embeddings) == 0:

        print(f"No valid embeddings found for {label}")
        continue

    average_embedding = np.mean(
        np.array(embeddings),
        axis=0
    )

    database.append(
        {
            "label": label,
            "embedding": average_embedding
        }
    )

    print(f"Used {len(embeddings)} embeddings.")

# ----------------------------------
# Save Database
# ----------------------------------

os.makedirs(
    "data/embeddings",
    exist_ok=True
)

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(database, f)

print("\n----------------------------------")
print("Average embedding database saved.")
print(f"Total identities: {len(database)}")
print("----------------------------------")