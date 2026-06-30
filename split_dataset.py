import os
import random
import shutil

# ----------------------------------
# Configuration
# ----------------------------------

SOURCE_DIR = "data/processed"
TRAIN_DIR = "data/processed/train"
TEST_DIR = "data/processed/test"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ----------------------------------
# Create Output Directories
# ----------------------------------

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Create unknown folder in test set
os.makedirs(
    os.path.join(TEST_DIR, "unknown"),
    exist_ok=True
)

# ----------------------------------
# Process Each Athlete
# ----------------------------------

for label in os.listdir(SOURCE_DIR):

    source_path = os.path.join(SOURCE_DIR, label)

    # Skip train/test folders if script is run again
    if label in ["train", "test"]:
        continue

    if not os.path.isdir(source_path):
        continue

    images = []

    for image in os.listdir(source_path):

        image_path = os.path.join(source_path, image)

        if os.path.isfile(image_path):
            images.append(image)

    random.shuffle(images)

    split_index = int(len(images) * TRAIN_RATIO)

    train_images = images[:split_index]
    test_images = images[split_index:]

    train_label_dir = os.path.join(TRAIN_DIR, label)
    test_label_dir = os.path.join(TEST_DIR, label)

    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)

    # Copy Training Images
    for image in train_images:

        shutil.copy2(
            os.path.join(source_path, image),
            os.path.join(train_label_dir, image)
        )

    # Copy Testing Images
    for image in test_images:

        shutil.copy2(
            os.path.join(source_path, image),
            os.path.join(test_label_dir, image)
        )

    print(f"{label}")
    print(f"  Train : {len(train_images)}")
    print(f"  Test  : {len(test_images)}")
    print("-" * 35)

print("\nDataset split completed successfully.")