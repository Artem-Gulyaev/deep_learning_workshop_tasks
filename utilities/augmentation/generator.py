#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
from numpy.linalg import inv
import imageio
import glob
import json
import os
import math

def init(configPath):
    with open(configPath) as jsonData:
        data = json.load(jsonData)
        try:
            print('Accepted following configuration: scale({0}...{1}), '
            'rotation({2}...{3}), input:{4}, output:{5}, clones: {6},'
            'out size: {7}x{8}'
            .format(data['scale']['min'],data['scale']['max'],
            data['rotation']['from'],data['rotation']['to'],
            data['input_path'],data['output_path'],data['clones_amount'],
            data['output_shape']['height'],data['output_shape']['width']))
        except KeyError as err:
            print('Wrong or missing parameter in config file:{0}'.format(err))
            return json.loads('{}')
        return data

def readImage(imagePath, format):
    image = imageio.imread(imagePath, format)
    l = len(image.shape)
    outImage = np.zeros((image.shape[0], image.shape[1]))
    for y in range(0, image.shape[0]):
        for x in range(0, image.shape[1]):
            if l > 2:
                c = np.sum(image[y,x,0:3]) / 3.0
            else:
                c = image[y,x]
            if c > 128:
                outImage[y,x] = 255
    return outImage

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
        clone = np.empty((config['output_shape']['height'],
        config['output_shape']['width']))
        clone.fill(255)
        alpha = np.random.uniform(config['rotation']['from'],
        config['rotation']['to'])

        alpha = alpha / 180.
        c,s = np.cos(alpha), np.sin(alpha)
        rotationMatrix = np.array([[c,-s,0],[s,c,0],[0,0,1]])

        scale = np.random.uniform(config['scale']['min'], config['scale']['max'])
        scaleMatrix = np.array([[scale,0,0],[0,scale,0],[0,0,1]])

        shiftXMax = clone.shape[1] * (1.0 - scale)
        shiftYMax = clone.shape[0] * (1.0 - scale)
        shift_x = np.random.uniform(-shiftXMax, shiftXMax)
        shift_y = np.random.uniform(-shiftYMax, shiftYMax)
        shiftMatrix = np.array([[1,0,shift_y/clone.shape[0]],[0,1,shift_x/clone.shape[1]],[0,0,1]])

        transformationMatrix = inv(scaleMatrix.dot(shiftMatrix).dot(rotationMatrix))

        for y in range(0, clone.shape[0]):
            for x in range(0, clone.shape[1]):
                destinationCoordinate = np.array([y/clone.shape[0] - 0.5, x/clone.shape[1] - 0.5, 1])

                delta = np.array(((destinationCoordinate > 0) - 0.5) * 2 / scale)
                delta = np.divide(np.divide(image.shape, clone.shape), delta[0:2])

                sourceCoordinate = transformationMatrix.dot(destinationCoordinate)
                if (sourceCoordinate[0:2] < 0.5).all() and (sourceCoordinate[0:2] > -0.5).all():

                    s_1 = ((sourceCoordinate[0:2] + 0.5) * image.shape).astype(int)
                    s_2 = s_1 - delta.astype(int)
                    s_min = np.minimum(s_1, s_2)
                    s_max = np.maximum(s_1, s_2)

                    area = image[s_min[0] : 1 + s_max[0], s_min[1] : 1 + s_max[1]]
                    #raw_input()
                    clone[y,x] = np.amin(area)

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
                image = readImage(imagePath, 'PNG-PIL')
            elif extention == 'png':
                print('--JPEG: {0}'.format(fileName))
                image = readImage(imagePath, 'JPEG-PIL')
            elif extention == 'bmp':
                print('--BMP: {0}'.format(fileName))
                image = readImage(imagePath, 'BMP-PIL')
            else:
                continue

            counter += processImage(image, config, fileName)

    print('Created {0} images'.format(counter))

if __name__ == '__main__':
    main()
