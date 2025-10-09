import wing_damselfly as wing
import pandas as pd
import cv2
import numpy as np
from config import *
import os


# Load in image
photos_folder = "/home3/tmjj24/apps/wing_damselfly/photos_standard/2022-2023/"
photos_folder_output = "/home3/tmjj24/apps/wing_damselfly/photos_standard/2022-2023-edited/"

# Create output folder
if not os.path.exists(photos_folder_output ):
    os.makedirs(photos_folder_output )
    print(f"Folder '{photos_folder_output }' created at {photos_folder_output }")
else:
    print(f"Folder '{photos_folder_output }' already exists at {photos_folder_output}")


files = sorted(os.listdir(photos_folder), reverse=True)

for file in files:
    # Read in file
    print(file)
    image_path = os.path.join(photos_folder, file)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # Get image neam without extension
    image_name = file.split('.')[0]
    print(image_name)
    
# Alpha and beta parameters
    alpha = 1.5
    beta = 0

    new_image = np.clip(image * alpha + beta, 0, 255).astype(np.uint8)
    # Convert alpha and beta in chars and replace . with _
    alpha_str = str(alpha).replace('.', '-')
    beta_str = str(beta).replace('-', 'm')
    # beta_str = str(beta).replace('.', '-')
    # save image
    print(f"Saving image as {photos_folder_output}/{image_name}_{alpha_str}_{beta_str}.jpg")
    cv2.imwrite(f"{photos_folder_output}/{image_name}_{alpha_str}_{beta_str}.jpg", new_image)



