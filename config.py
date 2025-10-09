# bool value, set True forewing and hindwing will be processed separately, 
# set False both wings will be processed together
is_separate = True

# bool value, set True for perching photos, False for standard photos
is_perching = False

# string value, set 'rembg' to use rembg u2net to remove photo backgrounds
# set 'detectron2' to use detectron2 model
used_model = 'detectron2' 

# the path of detectron2 model 
model_path = './models/standard_model_final.pth'
# model_path = './models/perching_model_final.pth'
model_config = './models/model_config.yaml'

# base photo folder, where all photo-related folders are located
# Base folder for iNat ./iNat/photos-original-2023-11-20-curated
base_photo_folder = './photos_standard/'

# you need to create report_file_path folder and csv file 
# and create photos_folder folder by yourself 
# if you create a blank base_photo_folder folder

# the path of report csv file 
report_file_path = f'{base_photo_folder}report/standard_wing_2022-2023-editied-A1-5B0.csv'

# folder with photos to be processed, put your photos in this folder
photos_folder = f'{base_photo_folder}2022-2023-edited'

# folder below can be auto created by program 

# folder with photos of identified and labeled wings
photos_detectron2_recognizer_path = f'{base_photo_folder}detectron2_recognizer'

# folder with segmented and background-removed wing photos (detectron2 model)
photos_detectron2_segmenter_path = f'{base_photo_folder}detectron2_segmenter'

# folder with segmented and background-removed wing photos (rembg u2net)
photos_rembg_segmenter_path = f'{base_photo_folder}rembg_segmenter'

# folder with photos of spots separated from the wings
photos_extractor_path = f'{base_photo_folder}extractor'

# Test folder for christophe to dump images
test_image_output = f'{base_photo_folder}test_dump'