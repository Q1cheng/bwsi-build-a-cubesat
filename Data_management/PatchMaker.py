import os
import cv2

input_folder = "input_images"
output_folder = "patches_gray"

patch_width = 2048
patch_height = 4092

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):

    if filename.lower().endswith((".png")):

        path = os.path.join(input_folder, filename)

        img = cv2.imread(path)

        if img is None:
            continue

        height, width, _ = img.shape

        patch_id = 0

        for y in range(0, height - patch_height + 1, patch_height):
            for x in range(0, width - patch_width + 1, patch_width):

                patch = img[y:y+patch_height, x:x+patch_width]

                # convert to grayscale
                gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

                out_name = f"{os.path.splitext(filename)[0]}_{patch_id}.png"
                out_path = os.path.join(output_folder, out_name)

                cv2.imwrite(out_path, gray_patch)

                patch_id += 1

print("Finished cropping and converting to grayscale.")