import os
from argparse import ArgumentParser
import cv2
import numpy as np

def parse_boxes(tray_path):
    result = []
    with open(tray_path+'/bounding_boxes/food_image_bounding_box.txt', 'rt') as f:
        for line in f.readlines():
            if line.startswith('ID: '):
                food_type, box = line.removeprefix('ID: ').split('; ')
                food_type, box = int(food_type), eval(box)
                result.append((food_type, box))
    return result

def crop_tray_foods(tray_path):
    image = cv2.imread(tray_path+'/food_image.jpg')
    box_images = []
    parsed_boxes = parse_boxes(tray_path)

    for food_type, box in parsed_boxes:
        x,y,w,h = box
        box_image = image[y:y+h, x:x+w]
        box_images.append((food_type, box, box_image))
    
    return box_images

                

def main():
    DATA_PATH = '../../data'
    BOW_DICTIONARY_PATH = "../../data/bow_dictionary"
    trays_path = filter(lambda p: p.startswith('tray'), os.listdir(DATA_PATH))
    
    for tray in trays_path:
        tray_path = os.path.join(DATA_PATH, tray)
        for food_type, box, food_image in crop_tray_foods(tray_path):
            file_title = f'{food_type}_{tray}.jpg'
            cv2.imwrite(os.path.join(BOW_DICTIONARY_PATH, file_title), food_image)

if __name__ == '__main__':
    main()
