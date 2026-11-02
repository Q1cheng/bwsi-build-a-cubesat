import os
import cv2

# Input and output folders
input_folder = "input_images"
output_folder = "grayscaled"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    
    if filename.lower().endswith(".png"):
        
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        image = cv2.imread(input_path)

        if image is None:
            print(f"Failed to load {filename}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(output_path, gray)

        print(f"Processed: {filename}")

print("Done.")