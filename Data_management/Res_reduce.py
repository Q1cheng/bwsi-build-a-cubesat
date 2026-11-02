import os
import cv2

input_folder = "input_images"
output_folder = "resized_images"

target_width = 720
target_height = 1072

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):

    if filename.lower().endswith((".png")):

        path = os.path.join(input_folder, filename)

        img = cv2.imread(path)

        if img is None:
            continue

        resized = cv2.resize(
            img,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA
        )

        out_path = os.path.join(output_folder, filename)

        cv2.imwrite(out_path, resized)

print("All images resized.")