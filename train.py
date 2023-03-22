import zipfile
import datetime
import string
import glob
import math
import os

import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.model_selection
import keras_ocr_3
import argparse
import cv2
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--Dataset',
                        default=None,
                        help='Path to data folder (train and validation in specific format)')
    parser.add_argument('--workdir', default = 'workdir',help='The dir to save logs and models')
    parser.add_argument(
        '--resume',
         default=None,
        help='path to checkpoint file')
    parser.add_argument(
        '--backbone',
        choices=['vgg', 'EfficientNetB0','EfficientNetB1','EfficientNetB2','EfficientNetB3','EfficientNetB4','EfficientNetB5'],
        default='vgg',
        help='backbone of model')
    parser.add_argument(
        '--lr',
        default=0.001,
        help='learning rate')
    parser.add_argument(
        '--batchsize',
        default=1,
        help='batchsize')
    parser.add_argument(
        '--epoch',
        default=100,
        help='number of epoch')
    parser.add_argument(
        '--save_gen_dataset',
        default=None,
        help='folder to save auto gen dataset')
    args = parser.parse_args()
    return args
def auto_gen_dataset(dataset,number,type, out_folder):
    isExist = os.path.exists(out_folder+'/'+type+'/imgs/')
    if not isExist:
        os.makedirs(out_folder+'/'+type+'/imgs/')
    isExist = os.path.exists(out_folder+'/'+type+'/gt/')
    if not isExist:
        os.makedirs(out_folder+'/'+type+'/gt/')
    i = 0
    while i <= number:
            img , label = next(dataset)  # tuong ung vs 1 anh
            i = i+1
            cv2.imwrite(out_folder+'/'+type+'/imgs/'+(str)(i)+'.png',img)
            with open(out_folder+'/'+type+'/gt/'+(str)(i)+'.txt', 'w') as f:
                string_news = []
                for lab in label:  # tung doi tuong trong label
                    for hehe in lab:
                        string_new = (str)(hehe[0][0][0]) +','+(str)(hehe[0][0][1])+','+(str)(hehe[0][1][0])+','+(str)(hehe[0][1][1])+','+(str)(hehe[0][2][0])+','+(str)(hehe[0][2][1])+','+(str)(hehe[0][3][0])+','+(str)(hehe[0][3][1])+','+hehe[1]
                        string_news.append(string_new +'\n')
                f.writelines(string_news)   
def create_image_generators(img_folder):
    """
        box is an array of points with shape (4, 2) providing the coordinates
        of the character box in clockwise order starting from the top left.
    """
    labels = []
    imgs = []
    for i in os.listdir(img_folder+'/images/')[0:200]:
        try:
            label = []
            img = cv2.imread(img_folder+'/images/'+i)
            imgs.append(img)
            split_ = i.split('.')[0]
            file = open(img_folder+'/gt/'+split_+'.txt', 'r')
            Lines = file.readlines()
            for line in Lines:
                line = line.strip()
                numbers = line.split(',')
                array = np.array([[(float)(numbers[0]),(float)(numbers[1])],[(float)(numbers[2]),(float)(numbers[3])],[(float)(numbers[4]),(float)(numbers[5])],[(float)(numbers[6]),(float)(numbers[7])]])
                element = [(array, numbers[8])]
                label.append(element)
            labels.append(label)
            yield img, label
        except:
            print('')
def main():
    args = parse_args()
    detector = keras_ocr_3.detection.Detector(weights=args.resume, backbone_name=args.backbone)
    detector_batch_size = args.batchsize
    isExist = os.path.exists(args.workdir)
    if not isExist:
        os.makedirs(args.workdir)
    isExist = os.path.exists(args.workdir+'/logs')
    if not isExist:
        os.makedirs(args.workdir+'/logs')
    
    detector_basepath = os.path.join(args.workdir, f'detector_')
    if args.Dataset == None:
            alphabet = string.digits + string.ascii_letters + '!?. '
            recognizer_alphabet = ''.join(sorted(set(alphabet.lower())))
            isExist = os.path.exists('dataset')
            if not isExist:
                os.makedirs('dataset')
            fonts = keras_ocr_3.data_generation.get_fonts(
                alphabet=alphabet,
                cache_dir='dataset'
            )
            backgrounds = keras_ocr_3.data_generation.get_backgrounds(cache_dir='dataset')
            text_generator = keras_ocr_3.data_generation.get_text_generator(alphabet=alphabet)

            def get_train_val_test_split(arr):
                train, valtest = sklearn.model_selection.train_test_split(arr, train_size=0.8, random_state=42)
                val, test = sklearn.model_selection.train_test_split(valtest, train_size=0.5, random_state=42)
                return train, val, test

            background_splits = get_train_val_test_split(backgrounds)
            font_splits = get_train_val_test_split(fonts)

            image_generators = [
                keras_ocr_3.data_generation.get_image_generator(
                    height=640,
                    width=640,
                    text_generator=text_generator,
                    font_groups={
                        alphabet: current_fonts
                    },
                    backgrounds=current_backgrounds,
                    font_size=(60, 120),
                    margin=50,
                    rotationX=(-0.05, 0.05),
                    rotationY=(-0.05, 0.05),
                    rotationZ=(-15, 15)
                )  for current_fonts, current_backgrounds in zip(
                    font_splits,
                    background_splits
                )
            ]  
            if args.save_gen_dataset !=None:
                 auto_gen_dataset(image_generators[0],len(background_splits[0]),'Train',args.save_gen_dataset)
                 auto_gen_dataset(image_generators[1],len(background_splits[1]),'Val',args.save_gen_dataset)
                 auto_gen_dataset(image_generators[2],len(background_splits[2]),'Test',args.save_gen_dataset)
            print('Dataset created!')
            detection_train_generator, detection_val_generator, detection_test_generator = [detector.get_batch_generator(
                    image_generator=image_generator,
                    batch_size=detector_batch_size
                ) for image_generator in image_generators
            ]    
            len_train =   len(background_splits[0])
            len_val =    len(background_splits[1])
            print('Len train and val ',len_train, len_val)
    else:
            train_generator = create_image_generators(args.Dataset+ '/Train/')
            len_train = len(os.listdir(args.Dataset+ '/Train/gt/'))
            detection_train_generator = detector.get_batch_generator(
                    image_generator=train_generator,
                    batch_size=detector_batch_size
                )
            val_generator = create_image_generators(args.Dataset+ '/Val/')
            len_val = len(os.listdir(args.Dataset+ '/Val/gt/'))
            detection_val_generator= detector.get_batch_generator(
                    image_generator=val_generator,
                    batch_size=detector_batch_size
                )
    def printCustom(batch,logs):
        with open(args.workdir+'/customlogs.txt','a+') as f:
                    f.write(f'batch is {batch} \n')
                    f.write(f'logs {logs} \n')
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    # define model
    detector.model.compile(loss="mse", optimizer=opt, metrics=['accuracy'])
    history = detector.model.fit_generator(
    generator=detection_train_generator,
    steps_per_epoch=math.ceil( len_train/ detector_batch_size),
    epochs=args.epoch,
    workers=0,
    callbacks=[
        tf.keras.callbacks.LambdaCallback(on_batch_end = printCustom),
        tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=5),
        tf.keras.callbacks.CSVLogger(f'{detector_basepath}.csv'),
         tf.keras.callbacks.ModelCheckpoint(filepath=f'{detector_basepath}.h5')
        ],
    validation_data=detection_val_generator,
    validation_steps=math.ceil(len_val/ detector_batch_size))        
    return history
if __name__ == '__main__':
    main()

