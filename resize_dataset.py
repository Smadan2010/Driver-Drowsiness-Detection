import os
import cv2

# Dataset location
DATASET_PATH = "./dataset/train"

# Your four classes
CLASS_NAMES = [
    "Closed",
    "Open",
    "no_yawn",
    "yawn"
]

# Target image size
TARGET_SIZE = (224, 224)

print("=" * 60)
print("RESIZING DATASET IMAGES")
print("=" * 60)

total = 0
resized = 0
failed = 0

for class_name in CLASS_NAMES:

    class_path = os.path.join(DATASET_PATH, class_name)

    if not os.path.exists(class_path):
        print(f"\n❌ Folder not found: {class_path}")
        continue

    print(f"\nProcessing: {class_name}")

    for filename in os.listdir(class_path):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):
            continue

        image_path = os.path.join(class_path, filename)

        total += 1

        # Read image
        image = cv2.imread(image_path)

        if image is None:
            print(f"❌ Could not read: {filename}")
            failed += 1
            continue

        # Resize to 224 × 224
        resized_image = cv2.resize(
            image,
            TARGET_SIZE,
            interpolation=cv2.INTER_AREA
        )

        # Save back to the same file
        success = cv2.imwrite(image_path, resized_image)

        if success:
            resized += 1
        else:
            print(f"❌ Could not save: {filename}")
            failed += 1

print("\n" + "=" * 60)
print("RESIZING COMPLETED")
print("=" * 60)

print(f"Total images found    : {total}")
print(f"Successfully resized  : {resized}")
print(f"Failed                : {failed}")

print("\nAll successfully processed images are now 224 × 224.")