import wing_damselfly as wing
import pandas as pd
import cv2
import numpy as np
from config import *
import os


# Load in image
input_csv = "/home3/tmjj24/data/standard_photos/CR2022_WingImageMeasurements_Jingyi_updated.csv"
photos_folder = "/home3/tmjj24/apps/wing_damselfly/photos_standard/my_photos_R/"
photos_folder_output = "/home3/tmjj24/apps/wing_damselfly/photos_standard/my_photos_edited_all/"

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
    #  
    # alpha = 1.0
    # beta = 0
    # new_image = np.clip(image * alpha + beta, 0, 255).astype(np.uint8)
    # alpha_str = str(alpha).replace('.', '')
    # beta_str = str(beta).replace('-', '')
    # # save image
    # print(f"Saving image as {photos_folder_output}/{image_name}_{alpha_str}_{beta_str}.png")
    # cv2.imwrite(f"{photos_folder_output}/{image_name}_{alpha_str}_{beta_str}.png", new_image)



    # Equalise histogram
    #Convert to grayscale
    #img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    ## equalize the histogram of the Y channel
    #img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    ## convert the YUV image back to RGB format
    #new_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    #print(f"Saving image as {photos_folder_output}/{image_name}_equalhist.png")
    #cv2.imwrite(f"{photos_folder_output}/{image_name}_equalhist.png", new_image)
#
    ## Equal histogram and then contrast
    #new_image = np.clip(image * alpha + beta, 0, 255).astype(np.uint8)
    #print(f"Saving image as {photos_folder_output}/{image_name}_equalhist.png")
    #cv2.imwrite(f"{photos_folder_output}/{image_name}_equalhist-{alpha_str}-{beta_str}.png", new_image)

    # For each photo run a range of alpha and beta conversions
    for alpha in np.arange(1.0, 2.1, 0.25):
        for beta in range(-200, 200, 100):
            # manuipulate image
            # new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            new_image = np.clip(image * alpha + beta, 0, 255).astype(np.uint8)
            # Convert alpha and beta in chars and replace . with _
            alpha_str = str(alpha).replace('.', '-')
            beta_str = str(beta).replace('-', 'm')
            # beta_str = str(beta).replace('.', '-')
            # save image
            print(f"Saving image as {photos_folder_output}/{image_name}_{alpha_str}_{beta_str}.png")
            cv2.imwrite(f"{photos_folder_output}/{image_name}_{alpha_str}_{beta_str}.png", new_image)



