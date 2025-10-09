import os
import recognizer
import segmenter
import extractor
import exporter
from config import *
import cv2
from detectron2.data import MetadataCatalog


# main logical
def process_photos():
    # load pre-trained model
    predictor, cfg = recognizer.load_detectron2_model(model_path)
    files = sorted(os.listdir(photos_folder), reverse=True)
    check_path()


    for file in files:
        image_name = file.split('.')[0]
        print("Starting analsysis on ", image_name, " or ", file)
        image_path = os.path.join(photos_folder, file)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        # recognize(predict) image (recognizer.py)
        prediction, output_image = recognizer.make_prediction(predictor, cfg, image)

        print(image_name, '==============')
        # Calculate the number of objects recognized

        if(not exist_objective(prediction)):
            csv_data = [image_name, 'No objects recognized']
            exporter.add_report_data(csv_data)
            continue
        
        # Calculate the number of objects found, if more than 1 proceed otherwise skip file
        masks = prediction["instances"].pred_masks.cpu().numpy()
        print(f"The predictor function found {masks.shape[0]} objects")
        
        if(masks.shape[0] < 2 and is_separate):
            print(f"Predictor found less than one object, skipping file")
            csv_data = [image_name, 'Less than 2 objects recognized']
            exporter.add_report_data(csv_data)
            continue

        save_image(output_image, image_name, photos_detectron2_recognizer_path)
        
        # segment wing (segmenter.py and recognizer.py)
        segmenter_path = photos_detectron2_segmenter_path if used_model == 'detectron2' else photos_rembg_segmenter_path
        if(not is_separate):
            segmented_image = segment_wing(prediction, image)
            save_image(segmented_image, image_name, segmenter_path)
        else:
            hw_index, fw_index = recognizer.cal_hw_fw_index(prediction)
            segmented_hw_image = segment_wing(prediction, image, mode='separate', index=hw_index)
            save_image(segmented_hw_image, image_name+'_hw', segmenter_path)
            segmented_fw_image = segment_wing(prediction, image, mode='separate', index=fw_index)
            save_image(segmented_fw_image, image_name+'_fw', segmenter_path)

        # calculate area ratio (extractor.py)
        if(not is_separate):
            black_img, red_img, tip_img, has_red, has_tip, csv_data = extractor.spot_exe(segmented_image)
            save_image(black_img, image_name+'_black', photos_extractor_path)
            if(has_red) : save_image(red_img, image_name+'_red', photos_extractor_path)
            if(has_tip) : save_image(tip_img, image_name+'_tip', photos_extractor_path)

            csv_data.insert(0, image_name)

            exporter.add_report_data(csv_data)
        else:
            hw_black_img, hw_red_img, hw_tip_img, hw_has_red, hw_has_tip, hw_csv_data = extractor.spot_exe(segmented_hw_image, mode='hw')
            fw_black_img, fw_red_img, fw_tip_img, fw_has_red, fw_has_tip, fw_csv_data = extractor.spot_exe(segmented_fw_image, mode='fw')

            save_image(hw_black_img, image_name+'_hw_basal', photos_extractor_path)
            if(hw_has_red) : save_image(hw_red_img, image_name+'_hw_red', photos_extractor_path)
            if(hw_has_tip) : save_image(hw_tip_img, image_name+'_hw_tip', photos_extractor_path)
            save_image(fw_black_img, image_name+'_fw_basal', photos_extractor_path)
            if(fw_has_red) : save_image(fw_red_img, image_name+'_fw_red', photos_extractor_path)
            if(fw_has_tip) : save_image(fw_tip_img, image_name+'_fw_tip', photos_extractor_path)

            csv_data_hfw = hw_csv_data + fw_csv_data
            csv_data_hfw.insert(0, image_name)
            exporter.add_report_data(csv_data_hfw)


def segment_wing(prediction, image, mode='pair', index=0):
    if(used_model == 'detectron2'):
        detectron2_mask_image = segmenter.mask_wings(prediction, image, mode=mode, index=index)
        cropped_image = segmenter.crop_image(prediction, detectron2_mask_image, mode=mode, index=index)
        return cropped_image
    elif(used_model == 'rembg'):
        cropped_image = segmenter.crop_image(prediction, image, mode=mode, index=index)
        rembg_image = segmenter.rembg_wings(cropped_image)
        return rembg_image
    else:
        print('paramater error')


def save_image(image, image_name, output_path):
    cv2.imwrite(f"{output_path}/{image_name}.png", image)

def check_path():
    check_create_directory(photos_detectron2_recognizer_path)
    check_create_directory(photos_detectron2_segmenter_path)
    check_create_directory(photos_rembg_segmenter_path)
    check_create_directory(photos_extractor_path)
    check_create_directory(test_image_output)

def check_create_directory(directory_path):  
    if not os.path.exists(directory_path):  
        os.makedirs(directory_path)

def exist_objective(prediction):
    if len(prediction["instances"].pred_masks.cpu().numpy()) > 0:
        return True
    else:
        return False
