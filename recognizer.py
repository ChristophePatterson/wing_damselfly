from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode, GenericMask
from detectron2.engine import DefaultPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt
from rembg import new_session, remove
from config import *

# 读取预训练好的模型
def load_detectron2_model(model_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_config)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 
    predictor = DefaultPredictor(cfg)
    return predictor, cfg

def make_prediction(predictor, cfg, image):
    prediction = predictor(image)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    output_image = visualize_predictions(image, prediction, metadata)
    return prediction, output_image

# 获得推理完成后的图形（RGB array格式）
def visualize_predictions(image, prediction, metadata):
    v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.2)
    v = v.draw_instance_predictions(prediction["instances"].to("cpu"))
    return v.get_image()[:, :, ::-1]

# 获得轮廓(contour)mask，黑白照片,仅做验证
def get_contour(prediction, image):
    masks = prediction["instances"].pred_masks.cpu().numpy()
    num_instances = len(prediction["instances"])

    for i in range(num_instances):
        contour_image = np.zeros_like(image)
        
        # Set the contour region to white
        contour_image[masks[i]] = [255, 255, 255]
        
        # Save the contour image (you can choose your desired format)
        cv2.imwrite(f"contour_instance_{i}.png", contour_image)


def calculate_centroid(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None
    
    centroid_x = int(M["m10"] / M["m00"])
    centroid_y = int(M["m01"] / M["m00"])
    return centroid_x, centroid_y

    


def cal_hw_fw_index(prediction):
    masks = prediction["instances"].pred_masks.cpu().numpy()

    # 根据质心位置判断 hindwing 还是 forewing
    centroid_0 = calculate_centroid(masks[0])
    centroid_1 = calculate_centroid(masks[1])
    # if centroid[1] < masks[0].shape[0] / 2:
    if centroid_0[1] <= centroid_1[1]:
        hw = 0
        fw = 1
    else:
        hw = 1
        fw = 0  
    return hw, fw