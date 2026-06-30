from ultralytics import YOLO
import cv2
from pathlib import Path
from tqdm import tqdm

# -----------------------------
# Paths
# -----------------------------
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Create processed directory
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load YOLO Face Model
# -----------------------------
model = YOLO("models/yolo/yolov8n-face.pt")   # We'll download this model

# -----------------------------
# Supported Image Extensions
# -----------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# -----------------------------
# Process Every Celebrity Folder
# -----------------------------
for celebrity_folder in RAW_DIR.iterdir():

    if not celebrity_folder.is_dir():
        continue

    output_folder = PROCESSED_DIR / celebrity_folder.name
    output_folder.mkdir(exist_ok=True)

    image_count = 0

    image_files = [
        f for f in celebrity_folder.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    for image_path in tqdm(image_files, desc=celebrity_folder.name):

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        results = model(image)[0]

        face_index = 0

        for box in results.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            face = image[y1:y2, x1:x2]

            if face.size == 0:
                continue

            filename = output_folder / f"{image_count}_{face_index}.jpg"

            cv2.imwrite(str(filename), face)

            face_index += 1

        image_count += 1

print("Finished Processing Dataset.")