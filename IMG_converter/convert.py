import os
import rasterio
import numpy as np
from PIL import Image

input_folder = "training_images"

# send output to Desktop
output_folder = os.path.expanduser("~/Desktop")

for filename in os.listdir(input_folder):

    if filename.lower().endswith(".img"):

        img_path = os.path.join(input_folder, filename)

        with rasterio.open(img_path) as src:
            data = src.read(1)

        # normalize values to 0-255
        data = data.astype(np.float32)
        data = (data - data.min()) / (data.max() - data.min())
        data = (data * 255).astype(np.uint8)

        image = Image.fromarray(data)

        out_name = os.path.splitext(filename)[0] + ".png"
        out_path = os.path.join(output_folder, out_name)

        image.save(out_path)

        print("Converted:", filename)