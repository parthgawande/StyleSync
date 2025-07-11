import os

image_dir = "../data/images"
dir_count = 0
image_count = 0

for root, dirs, files in os.walk(image_dir):
    if root == image_dir:
        dir_count += len(dirs)
    image_count += sum(1 for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png')))

print(f"Total directories (outfits): {dir_count}")
print(f"Total images: {image_count}")
