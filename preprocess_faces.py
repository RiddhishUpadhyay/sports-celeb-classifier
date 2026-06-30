from pathlib import Path
import cv2
from tqdm import tqdm

# -------------------------
# Configuration
# -------------------------
INPUT_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/preprocessed")

IMAGE_SIZE = (112, 112)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Process Each Celebrity Folder
# -------------------------
for celebrity_folder in INPUT_DIR.iterdir():

    if not celebrity_folder.is_dir():
        continue

    output_folder = OUTPUT_DIR / celebrity_folder.name
    output_folder.mkdir(parents=True, exist_ok=True)

    image_files = [
        img for img in celebrity_folder.iterdir()
        if img.suffix.lower() in IMAGE_EXTENSIONS
    ]

    for image_path in tqdm(image_files, desc=celebrity_folder.name):

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read: {image_path}")
            continue

        # Convert BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize
        image = cv2.resize(image, IMAGE_SIZE)

        # Convert back to BGR before saving
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        output_path = output_folder / image_path.name

        cv2.imwrite(str(output_path), image)

print("\nPreprocessing Complete!")