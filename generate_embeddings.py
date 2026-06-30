from pathlib import Path
import cv2
import pickle
from tqdm import tqdm
from insightface.app import FaceAnalysis
import os

# -----------------------------
# Paths
# -----------------------------

# INPUT_DIR = Path("data/preprocessed")
# OUTPUT_DIR = Path("data/embeddings")
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# OUTPUT_FILE = OUTPUT_DIR / "face_database.pkl"

# -----------------------------
# Load ArcFace Model
# -----------------------------
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=0)

# ----------------------------------
# Paths
# ----------------------------------

TRAIN_DIR = "data/processed/train"
OUTPUT_PATH = "data/embeddings/face_database.pkl"


database = []

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# -----------------------------
# Generate Embeddings
# -----------------------------
# for celebrity_folder in INPUT_DIR.iterdir():

#     if not celebrity_folder.is_dir():
#         continue

#     label = celebrity_folder.name

#     image_files = [
#         img for img in celebrity_folder.iterdir()
#         if img.suffix.lower() in IMAGE_EXTENSIONS
#     ]

#     for image_path in tqdm(image_files, desc=label):

#         image = cv2.imread(str(image_path))

#         if image is None:
#             continue

#         faces = app.get(image)

#         if len(faces) != 1:
#             continue

#         embedding = faces[0].embedding

#         database.append({
#             "label": label,
#             "embedding": embedding
#         })

for label in os.listdir(TRAIN_DIR):

    label_path = os.path.join(TRAIN_DIR, label)

    if not os.path.isdir(label_path):
        continue

    print(f"\nProcessing {label}...")

    image_count = 0

    for image_name in os.listdir(label_path):

        image_path = os.path.join(label_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
            print(f"Skipped: {image_name}")
            continue

        faces = app.get(image)
        # We expect exactly one face
        if len(faces) != 1:
            print(
                f"Skipped ({len(faces)} faces): {image_name}"
            )
            continue

        embedding = faces[0].embedding

        database.append(
            {
                "label": label,
                "image_name": image_name,
                "embedding": embedding
            }
        )

        image_count += 1

    print(f"Saved {image_count} embeddings.")

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
print(f"Database saved successfully!")
print(f"Total embeddings: {len(database)}")
print("----------------------------------")
# # -----------------------------
# # Save Database
# # -----------------------------
# with open(OUTPUT_FILE, "wb") as f:
#     pickle.dump(database, f)

# print(f"\nSaved {len(database)} embeddings.")
# print(f"Database saved to {OUTPUT_FILE}")