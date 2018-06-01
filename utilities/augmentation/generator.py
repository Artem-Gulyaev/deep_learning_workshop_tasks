#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import imageio
import glob
import json
import os

def init(configPath):
    with open(configPath) as jsonData:
        data = json.load(jsonData)
        try:
            print('Accepted following configuration: scale({0}...{1}), rotation({2}...{3}), input:{4}, output:{5}, clones: {6},out size: {7}x{8}'
            .format(data['scale']['min'],data['scale']['max'],data['rotation']['from'],data['rotation']['to'],
            data['input_path'],data['output_path'],data['clones_amount'],data['output_shape']['height'],data['output_shape']['width']))
        except KeyError as err:
            print('Wrong or missing parameter in config file:{0}'.format(err))
            return json.loads('{}')
        return data

def readPNGImage(imagePath):
    image = imageio.imread(imagePath)
    l = len(image.shape)
    outImage = np.zeros((image.shape[0],image.shape[1]))
    for y in range(0,image.shape[0]):
        for x in range(0,image.shape[1]):
            if l > 2:
                c = np.sum(image[y,x,0:3]) / 3.0
            else:
                c = image[y,x]
            if c > 128:
                outImage[y,x] = 255
    return outImage

def readJPEGImage(imagePath):
    return np.empty((0,0))

def readBMPImage(imagePath):
    return np.empty((0,0))

def writeImage(imagePath, image):
    #raw_input("Press Enter to continue...")
    imageio.imwrite(imagePath, image)

def checkPathAndCreatIfNotExist(path):
    if os.path.exists(path):
        return True
    try:
        os.makedirs(path)
    except OSError as err:
        return False
    return True

def processImage(image, config, fileName):
    counter = 0
    for i in range(0,config['clones_amount']):
        clone = np.empty((config['output_shape']['height'],config['output_shape']['width']))
        clone.fill(255)
        alpha = np.random.uniform(config['rotation']['from'],config['rotation']['to'])
        alpha = alpha / 180.
        c,s = np.cos(alpha), np.sin(alpha)
        R_matrix = np.array(((c,-s),(s,c)))
        scale = np.random.uniform(config['scale']['min'], config['scale']['max'])
        shiftXMax = clone.shape[1] * (1.0 - scale)
        shiftYMax = clone.shape[0] * (1.0 - scale)
        shift_x = np.random.uniform(-shiftXMax, shiftXMax)
        shift_y = np.random.uniform(-shiftYMax, shiftYMax)
        S_matrix = np.array([shift_y/clone.shape[0],shift_x/clone.shape[1]])
        for y in range(0, clone.shape[0]):
            for x in range(0, clone.shape[1]):
                sourceCoordinate = np.array([y/clone.shape[0] - 0.5, x/clone.shape[1] - 0.5])
                sourceCoordinate = sourceCoordinate / scale

                sourceCoordinate = sourceCoordinate + S_matrix
                sourceCoordinate = R_matrix.dot(sourceCoordinate.T)
                if (sourceCoordinate < 0.5).all() and (sourceCoordinate > -0.5).all():
                    s_y = int((sourceCoordinate[0] + 0.5) * image.shape[0])
                    s_x = int((sourceCoordinate[1] + 0.5) * image.shape[1])
                    clone[y,x] = image[s_y,s_x]

        counter += 1
        writeImage(config['output_path']+'/{0}_{1}.png'.format(fileName,i), clone)

    return counter

def main():
    config = init('./config.json')
    if len(config) != 0 and checkPathAndCreatIfNotExist(config['output_path']):
        counter = 0
        for imagePath in glob.glob(config['input_path']+'/*'):
            fileNameWithExtention = os.path.basename(imagePath)
            fileName,extention = fileNameWithExtention.split('.')
            if extention == 'png':
                print('--PNG: {0}'.format(fileName))
                image = readPNGImage(imagePath)
            elif extention == 'png':
                print('--JPEG: {0}'.format(fileName))
                image = readJPEGImage(imagePath)
            elif extention == 'bmp':
                print('--BMP: {0}'.format(fileName))
                image = readBMPImage(imagePath)
            else:
                continue

            counter += processImage(image, config, fileName)

    print('Created {0} images'.format(counter))

if __name__ == '__main__':
    main()
