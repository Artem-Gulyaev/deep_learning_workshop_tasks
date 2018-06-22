import numpy as np
import math
#import matplotlib.pyplot as plt

def createDataSet(img_w,img_h,m):
    nx = img_w * img_h
    X = np.zeros((m,nx))
    Y = np.zeros((1,m))

    for i in range(0,img_w):
        for j in range(0,img_h):
            if (i==0 or  j==0 or i==img_h-1 or j==img_h-1):
                X[0][i+j*img_w] = 1.0;
                X[1][i+j*img_w] = 1.0;
    X[1][0]= 0.0;
    X[1][img_w-1]= 0.0;
    X[1][img_w*img_h-1]= 0.0;
    X[1][img_w*img_h-img_w]= 0.0;
    Y[0][0]=1.0;
    Y[0][1]=1.0;
    print("Y=" ,Y)
    #print ("X")
    #for i in range (0,m):
    #    print (np.reshape(X[i],(img_w,img_h)))
       #print("X")
    #print("Y")
    #print (Y)
    return (X,Y)

def initFun(params,nx):
    W = np.random.randn(1,nx) ##/ nx
    for i in range(nx):
      W[0][i] = 0.1 #i*0.01
    B = np.zeros((1,1))
    params["W"] = W # weight vector
    params["B"] = B
    print ("W=",W)
    return (W,B)

def optimize(params,X,Y,learning_rate,epochs):
    W = params["W"]
    B = params["B"]
    cost_history = np.zeros((epochs,))
    for i in range(epochs):
      optimizationStep(params,X,Y,learning_rate)
      cost_history[i]= params["J"]
      print ("Epohc",str(i),"cost = ",cost_history[i])

    return (W,B,cost_history)

def optimizationStep(params, X,Y,learning_rate):
    W= params["W"]
    B= params["B"]
    print("AEX")
    Y_hat = forwardProp(params,X)

    backProp(params,X,Y,Y_hat)

    return (W,B)

def forwardProp(params,X):
    W= params["W"]
    B= params["B"]
    SNum = params["SNum"]
    # matrix row ~ saples
    Xr = np.transpose(X)
    # column - pixels
    #print ("W=",W)
    #print ("X=",X)
    #print ("Xr=",Xr)
    Z = np.dot(W,Xr) + B
    params["Z"]= Z
    print("Z = ",Z)
    Y_hat = sigmoid(Z)
    print ("Y_hat",Y_hat)
    return Y_hat

def sigmoid(X):
    return 1/ (1+ np.exp(-X))

def backProp(params,X,Y,Y_hat):
    J= costFun(Y,Y_hat)
    params["J"] = J
    Z = params["Z"]
    m = params["SNum"]
    print("m=",m)

def costFun(Y,Y_hat):
    losses = lossFucntioncrossEntropy(Y,Y_hat)
    print("Y=",Y)
    print("Losses", losses)
    cost = np.sum(losses)
    cost /= losses.size
    return cost

def lossFucntioncrossEntropy(Y,Y_hat):
    return  - Y * np.log(Y_hat) - (1 - Y) * np.log(1 - Y_hat)

def main():
    img_w = 4
    img_h = 4
    learning_rate = 0.01
    epochs = 1
    SNum = 4
    print("Neron   recognition start point")
    (X,Y)=createDataSet(img_w,img_h,SNum)
    params = {}
    params["SNum"]= SNum
    initFun(params,img_w*img_h)
    (W,B,cost_history) = optimize(params,X,Y,learning_rate,epochs)

    #print(params)


main()
