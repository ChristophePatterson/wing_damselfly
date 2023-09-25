from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode, GenericMask
from detectron2.engine import DefaultPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt
from rembg import new_session, remove

def mask_wings(prediction, image, mode='pair', index=0):
    original_image = image
    masks = prediction["instances"].pred_masks.cpu().numpy()
    binary_mask = np.zeros_like(masks[0], dtype=np.uint8)
    if(mode == 'pair'):
        for mask in masks:
            binary_mask = cv2.bitwise_or(binary_mask, mask.astype(np.uint8))
    elif(mode == 'separate'):
        binary_mask = cv2.bitwise_or(binary_mask, masks[index].astype(np.uint8))
    binary_mask = (binary_mask * 255).astype(np.uint8)
    wing_image = cv2.bitwise_and(original_image, original_image, mask=binary_mask.astype(np.uint8))
    b, g, r = cv2.split(wing_image)
    bgra = cv2.merge([b, g, r, binary_mask])
    return bgra

def rembg_wings(image_numpy):
    # u2net  isnet-general-use
    model_name = "u2net"
    session = new_session(model_name)
    output = remove(image_numpy, session=session)
    output = remove(output, session=session)
    # cv2.imwrite('test_rembg.png', output)
    return output


def crop_image(prediction, image, mode='pair', index=0):
    instances = prediction["instances"]
    boxes = instances.pred_boxes.tensor.cpu().numpy()
    range_number = index
    if(mode == 'pair'):
        x_min = np.min(boxes[:, 0])
        y_min = np.min(boxes[:, 1])
        x_max = np.max(boxes[:, 2])
        y_max = np.max(boxes[:, 3])
    elif((mode == 'separate')):
        x_min = np.min(boxes[range_number, 0])
        y_min = np.min(boxes[range_number, 1])
        x_max = np.max(boxes[range_number, 2])
        y_max = np.max(boxes[range_number, 3])
    # output_path = "test_crop_mask_image.png"
    image_cv_cropped = image[int(y_min):int(y_max), int(x_min):int(x_max), :]
    return image_cv_cropped
    # cv2.imwrite(output_path, image_cv_cropped)
    # PIL way
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = Image.fromarray(image)
    # cropped_image = image.crop((x_min, y_min, x_max, y_max))
    # cropped_image.save(output_path)