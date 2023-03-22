import zipfile
import datetime
import string
import glob
import math
import os
import keras_ocr_3
import numpy as np
import PIL
from PIL import Image
from PIL import ImageDraw
import argparse
import easyocr

def parse_args():
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('--DataTest',
                        default=None,
                        help='Path to data test folder')
    parser.add_argument('--model',
                        choices=['EasyOCR','Keras_VGG','Keras_EfficientnetB0',
                                 'Keras_EfficientNetB1','Keras_EfficientNetB2','Keras_EfficientNetB3','Keras_EfficientNetB4','Keras_EfficientNetB5'],
                         help='model name')
    parser.add_argument('--checkpoint', help='Path to checkpoint file')
    parser.add_argument('--outfolder', help='Path to output folder')

    args = parser.parse_args()
    return args
def draw_boxes(image, bounds, color='yellow', width=2):
            draw = ImageDraw.Draw(image)
            for bound in bounds:
                p0, p1, p2, p3 = bound[0]
                draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
            return image
def main():
    args = parse_args()
    if args.DataTest == None:
        return
    else:
        list_img = os.listdir(args.DataTest)
    
    if args.model== 'Keras_EfficientnetB0':
            detector = keras_ocr_3.detection.Detector(weights=None,backbone_name= 'EfficientnetB0')
            if args.checkpoint != None:
                detector.model.load_weights(args.checkpoint)
    if args.model=='Keras_EfficientnetB1':
            detector = keras_ocr_3.detection.Detector(weights=None,backbone_name= 'EfficientnetB1')
            if args.checkpoint != None:
                detector.model.load_weights(args.checkpoint)
    if args.model== 'Keras_EfficientnetB2':
            detector = keras_ocr_3.detection.Detector(weights=None,backbone_name= 'EfficientnetB2')
            if args.checkpoint != None:
                detector.model.load_weights(args.checkpoint)
    if args.model== 'Keras_EfficientnetB3':
            detector = keras_ocr_3.detection.Detector(weights=None,backbone_name= 'EfficientnetB3')
            if args.checkpoint != None:
                detector.model.load_weights(args.checkpoint)
    if args.model== 'Keras_EfficientnetB4':
            detector = keras_ocr_3.detection.Detector(weights=None,backbone_name= 'EfficientnetB4')
            if args.checkpoint != None:
                detector.model.load_weights(args.checkpoint)
    if args.model== 'Keras_EfficientnetB5':
            detector = keras_ocr_3.detection.Detector(weights=None,backbone_name= 'EfficientnetB5')
            if args.checkpoint != None:
                detector.model.load_weights(args.checkpoint)
    if args.model== 'Keras_VGG':
            detector = keras_ocr_3.detection.Detector(weights='clovaai_general')
            if args.checkpoint != None:
                detector.model.load_weights(args.checkpoint)
    if args.model== 'EasyOCR':
            detector = easyocr.Reader(['vi','en'])
            for img_path in list_img:
                im = Image.open(args.DataTest+'/'+img_path)
                bounds = detector.readtext(args.DataTest+'/'+img_path)
                result = draw_boxes(im, bounds)
                #cv2.imwrite("C:/Users/Admin/Desktop/MyCode/EasyOCR/Datasets/Test_out/"+i,result)
                result.save(args.outfolder+'/'+img_path)
            return
    for img_path in list_img:
            imtest = Image.open(args.DataTest+'/'+img_path)
            bounds = detector.detect(np.expand_dims(imtest,axis = 0))[0]
            draw = ImageDraw.Draw(imtest)
            for bound in bounds:
                p0, p1, p2, p3 = bound
                draw.line([*p0, *p1, *p2, *p3, *p0], fill='yellow', width=2)
            imtest.save(args.outfolder+'/'+img_path)        
        
if __name__ == '__main__':
    main()