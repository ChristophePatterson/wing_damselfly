import cv2
import numpy as np
from config import *


def spot_exe(image, mode='pair'):
    image_hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
    image_gray = gen_gray_image(image)
    image_black_opt = adjust_brightness_contrast(image)

    # 定义颜色阈值范围 BGR
    lower_red = np.array([0, 90, 90], dtype=np.uint8)
    upper_red = np.array([5, 255, 255], dtype=np.uint8)

    lower_black = np.array([0, 0, 0, 0], dtype=np.uint8)
    upper_black = np.array([10, 10, 10, 255], dtype=np.uint8)
    
    lower_tip_color, upper_tip_color = gen_tip_color(mode)

    # 定义形态学内核（可以根据实际情况调整大小）
    black_kernel = np.ones((1, 1), np.uint8)
    red_kernel = np.ones((2, 2), np.uint8)

    # 创建掩膜
    red_mask = cv2.inRange(image_hsv, lower_red, upper_red)
    if(is_perching):
        black_mask = cv2.inRange(image_gray, np.array([0], dtype=np.uint8), np.array([1], dtype=np.uint8))
    else:
        black_mask = cv2.inRange(image_black_opt, lower_black, upper_black)
    # black_opt_mask = cv2.inRange(image, lower_black_opt, upper_black_opt)
    # black_mask = cv2.add(black_mask, black_opt_mask)
    tip_mask = cv2.inRange(image, lower_tip_color, upper_tip_color)
    tip_mask_sub = cv2.inRange(image, lower_tip_color, upper_tip_color)

    # 使用内核扩大范围
    red_mask = cv2.dilate(red_mask, red_kernel, iterations=2)
    black_mask = cv2.dilate(black_mask, black_kernel, iterations=7)
    # tip_mask = cv2.dilate(tip_mask, kernel, iterations=1)
    # tip_mask_sub = cv2.dilate(tip_mask_sub, kernel, iterations=1)

    # 以 1/6 和 2/6 定位tip
    tip_mask_zero_1, tip_mask_zero_2 = gen_tip_mask_zero(mode, image)

    # 处理 tip mask 实际范围
    tip_mask = produce_tip_mask(tip_mask, tip_mask_zero_1)
    tip_mask_sub = produce_tip_mask(tip_mask_sub, tip_mask_zero_2)

    # 计算图像总面积
    total_area = np.sum(image[:, :, 3] > 0)

    wing_has_red = has_red_area(image, red_mask, total_area)
    wing_has_tip = has_tip_area(image, tip_mask, tip_mask_sub)

    black_mask = produce_black_mask(wing_has_tip, wing_has_red, black_mask, tip_mask, tip_mask_zero_1 , red_mask) 

    # 计算红色和黑色区域的面积
    red_area = np.sum(np.logical_and(red_mask == 255, image[:, :, 3] > 0))
    black_area = np.sum(np.logical_and(black_mask == 255, image[:, :, 3] > 0))
    tip_area = np.sum(np.logical_and(tip_mask == 255, image[:, :, 3] > 0))

    # 计算面积占比
    red_area_ratio = red_area / total_area
    black_area_ratio = black_area / total_area
    tip_area_ratio = tip_area / total_area

    # 标记红色和黑色区域
    red_region = cv2.bitwise_and(image, image, mask=red_mask)
    black_region = cv2.bitwise_and(image, image, mask=black_mask)
    tip_region = cv2.bitwise_and(image, image, mask=tip_mask)
    # merged_region = cv2.bitwise_and(image, image, mask=merged_mask)


    print(mode + '======')
    print("Percentage of black area: ", black_area_ratio)
    if(wing_has_red) : print("Percentage of red area: ", red_area_ratio)
    if(wing_has_tip) : print("Percentage of tip area：", tip_area_ratio)

    csv_data = [total_area, black_area, black_area_ratio, red_area if wing_has_red else 0, red_area_ratio if wing_has_red else 0, tip_area if wing_has_tip else 0, tip_area_ratio if wing_has_tip else 0,]

    return black_region, red_region, tip_region, wing_has_red, wing_has_tip, csv_data

def gen_tip_color(mode):
    if(mode == 'pair'):
        lower_black_tip = np.array([35, 65, 105, 0], dtype=np.uint8)
        upper_black_tip = np.array([100, 140, 190, 255], dtype=np.uint8)
    elif(mode == 'hw'):
        lower_black_tip = np.array([5, 5, 5, 0], dtype=np.uint8)
        upper_black_tip = np.array([150, 150, 150, 255], dtype=np.uint8) 
    elif(mode == 'fw'):
        lower_black_tip = np.array([30, 65, 105, 0], dtype=np.uint8)
        upper_black_tip = np.array([110, 140, 200, 255], dtype=np.uint8)
    return lower_black_tip, upper_black_tip

def gen_tip_mask_zero(mode, image):
    if(mode == 'pair'):
        mask_pair_1 = np.zeros_like(image, dtype=np.uint8)
        mask_pair_1[image.shape[0] - image.shape[0] // 2:, :image.shape[1] // 3, :] = 255

        mask_pair_2 = np.zeros_like(image, dtype=np.uint8)
        mask_pair_2[image.shape[0] - image.shape[0] // 2:, image.shape[1] // 3:- image.shape[1] // 2, :] = 255
        
        return mask_pair_1,mask_pair_2
    elif(mode == 'fw'):
        mask_FW_1 = np.zeros_like(image, dtype=np.uint8)
        mask_FW_1[::, :image.shape[1] // 10, :] = 255

        mask_FW_2 = np.zeros_like(image, dtype=np.uint8)
        mask_FW_2[::, image.shape[1] // 10:- image.shape[1] // 10 * 8, :] = 255
        
        return mask_FW_1, mask_FW_2
    elif(mode == 'hw'):
        mask_HW_1 = np.zeros_like(image, dtype=np.uint8)
        mask_HW_1[::, :image.shape[1] // 5, :] = 255

        mask_HW_2 = np.zeros_like(image, dtype=np.uint8)
        mask_HW_2[::, image.shape[1] // 7:- image.shape[1] // 7 * 5, :] = 255
        
        return mask_HW_1,mask_HW_2
    
def produce_tip_mask(tip_mask, mask_zero):
    tip_mask[np.where(mask_zero[:, :, 0] == 0)] = 0
    tip_mask[np.where(mask_zero[:, :, 1] == 0)] = 0
    tip_mask[np.where(mask_zero[:, :, 2] == 0)] = 0
    return tip_mask

def produce_black_mask(has_tip: bool, has_red: bool, black_mask, tip_mask, tip_mask_zero_1, red_mask):
    mask = black_mask
    if(has_tip):
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(tip_mask))
    if(has_red):
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(red_mask))
    # else:
    #     black_mask = cv2.add(black_mask, tip_mask)
    return mask

def has_tip_area(image, tip_mask, tip_mask_sub):
    if(is_perching):
        return False
    tip_mask_area = np.sum(np.logical_and(tip_mask == 255, image[:, :, 3] > 0))
    tip_mask_sub_area = np.sum(np.logical_and(tip_mask_sub == 255, image[:, :, 3] > 0))
    if(tip_mask_sub_area >= tip_mask_area* 0.7):
        return False
    else:
        return True

def has_red_area(image, red_mask, totoal_area):
    red_area = np.sum(np.logical_and(red_mask == 255, image[:, :, 3] > 0))
    if(red_area / totoal_area > 0.01):
        return True
    else:
        return False


def adjust_brightness_contrast(image, brightness= -200, contrast = 1.7):
    if(is_perching):
        brightness = -65
        contrast = 2.2
    else:
        brightness = -200
        contrast = 1.7
    adjusted_image = np.clip(image * contrast + brightness, 0, 255).astype(np.uint8)
    return adjusted_image

def gen_gray_image(image):
    image = adjust_brightness_contrast(image, brightness=-65, contrast=2.2)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_gray = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image_gray