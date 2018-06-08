import numpy as np
import math
#import matplotlib.pyplot as plt

def createDataSet(img_w,img_h,m):
    nx = img_w * img_h
    X = np.zeros((m,nx))
    Y = np.zeros((1,m))
    print ("X = ",X)
    print ("Y = ",Y)
    return (X,Y)

def initFun(params,nx):
    W = np.random.randn(1,nx) ##/ nx
    B = np.zeros((1,1))
    params["W"] = W # weight vector
    params["B"] = B
    return (W,B)


def main():
    img_w = 2
    img_h = 2
    trainigSamplesNum = 5
    print("Neron   recognition start point")
    createDataSet(img_w,img_h,trainigSamplesNum)
    params = {}
    initFun(params,img_w*img_h)
    print(params)


main()
