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
model_config = './models/model_config.yaml'

# base photo folder, where all photo-related folders are located
base_photo_folder = './photos_rembg/'

# the path of report csv file 
report_file_path = f'{base_photo_folder}report/standard_wing.csv'

# folder with photos to be processed, put your photos in this foler
photos_folder = f'{base_photo_folder}standard_photos'

# folder with photos of identified and labeled wings
photos_detectron2_recognizer_path = f'{base_photo_folder}detectron2_recognizer'

# folder with segmented and background-removed wing photos (detectron2 model)
photos_detectron2_segmenter_path = f'{base_photo_folder}detectron2_segmenter'

# folder with segmented and background-removed wing photos (rembg u2net)
photos_rembg_segmenter_path = f'{base_photo_folder}rembg_segmenter'

# folder with photos of spots separated from the wings
photos_extractor_path = f'{base_photo_folder}extractor'