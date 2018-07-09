import numpy as np
import math
import matplotlib.pyplot as plt

def drawCircle(x,y,r,data):
    nx,ny = data.shape
    I,J= np.ogrid[-x:nx-x,-y:ny-y]
    mask = I*I + J*J <=r*r
    data[mask] = 1.0
    #print (data)
    return data
def drawRectangle(x,y,r,data):
    nx,ny = data.shape
    I,J= np.ogrid[-x:nx-x,-y:ny-y]
    mask = I*I + J*J <=r
    data[mask] = 1.0
    print (data)
    return data

def genNonRandomTrainigSet(pmap):
    SZ = pmap["SZ"]
    LEN = SZ*SZ
    X= np.zeros((LEN,1))
    Y= np.zeros((1,1))
    X = np.c_[X,np.reshape(drawCircle(2,2,2,np.zeros((SZ,SZ))),(LEN))]
    Y = np.c_[Y,1]

    X = np.c_[X,np.reshape(drawCircle(2,2,1,np.zeros((SZ,SZ))),(LEN))]
    Y = np.c_[Y,1]

    X = np.c_[X,np.reshape(drawRectangle(2,2,4,np.zeros((SZ,SZ))),(LEN))]
    Y = np.c_[Y,0]

    X = np.c_[X,np.reshape(drawRectangle(1,1,2,np.zeros((SZ,SZ))),(LEN))]
    Y = np.c_[Y,0]

    return (X,Y)

def initParams(pmap):
    W = np.random.randn(1,pmap["SZ"]*pmap["SZ"])
    B = np.zeros((1,1))
    pmap["W"] = W
    pmap["B"] = B
    return (W,B)
def opt(pmap,X,Y):
    W = pmap["W"]
    B = pmap["B"]
    ephs = pmap["epochs"]
    costHist = np.zeros((ephs,))
    for i in range(ephs):
        costHist[i]= processEpoch(pmap,X,Y)
    print (costHist)
    return (W,B,costHist)
def processEpoch(pmap,X,Y):
    W = pmap["W"]
    B = pmap["B"]
    Yhat = forwardProp(pmap,X)

    print(Yhat)
    return 2
def sigmoid(X):
    return 1/(1+np.exp(-X))
def sigmoidDeriv(X):
    return sigmoid(X)*(1-sigmoid(X))
def forwardProp(pmap,X):
    W = pmap["W"]
    B = pmap["B"]
    print("W",W.size,X.size)
    Z = np.dot(W,X)+B
    Yhat = sigmoid(Z)
    return Yhat
def main():
    pmap = {}
    pmap["SZ"] = 10
    pmap["lrate"]= 0.01
    pmap["epochs"] = 1
    (X,Y) = genNonRandomTrainigSet(pmap)
    (W,B) = initParams(pmap)
    (W,B,costHist) = opt(pmap,X,Y)

main()
