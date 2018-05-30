import numpy as np
import h5py
import math
import matplotlib.pyplot as plt

"""
This is an example of regular logistic regression, which
will fit the following hypotesis onto the given data.
"""

def sigmoid(X):
    """
    Calculates the element-wise sigmoid of its argument:
        sigmoid(x) = 1 / (1 + exp(-x))
    Parameters:
        X    	- the numpy array of arbitrary size or a
                  single value.
    """
    return 1 / (1 + np.exp(-X))

def sigmoidDerivative(X):
    """
    Calculates the sigmoid derivative
            d\sigma/dx = \sigma(x)(1 - \sigma(x))
    for given set of X.
    Parameters:
        X 		- numpy array or single value of point(s) where
                  to calculate the derivative.
    """

    sigma = sigmoid(X)

    # note, that "*" means element-wise product in numpy
    return sigma * (1 - sigma)

def forwardPropagationLogisticRegression(params, X):
    """
    Parameters:
        params 	- the dictionary containing the relevant parameters,
                  params["W"] - the weight matrix of the neuron,
                        [float{1,nx}].
                  params["B"] - the bias of the neuron,
                        [float{1,1}].
        X		- [float{nx,m}] the horizontal stack of vectors
                  (column vectors) of input shape(X) = (nx, m),
                  where m is a number of examples and nx is an
                  input vector size.
    Returns:
        Y_hat   - [float{1, m}] the predicted value of the
                  logistic regression for given X.
    """

    # W in this case is a transponed basis vector of the single
    #	neuron:
    #			(w_1 w_2 ... w_nx)
    # shape(W) == (1, nx)
    # where nx is a number of incoming connections from previous layer
    # of the network.
    W = params["W"]

    # B is a bias value of the single neuron
    # shape(B) == (1, 1)
    B = params["B"]

    # the single neuron conversion from input data vector to
    # single output value (or from matrix to row-vector)
    Z = np.dot(W,X) + B
    Y_hat = sigmoid(Z)

    # saving intermediate data to avoid excessive computations
    # on backward propagation step
    params["Z"] = Z
    params["Y_hat"] = Y_hat

    return Y_hat

def lossFunctionCrossEntropy(Y, Y_hat):
    """
    Calculates the cross-entropy loss function for given correct
    answer y and predicted value y_hat.
    Parameters:
        Y 		- [float: ny x m] the correct answer
        Y_hat   - [float: ny x m] the answer prediction
    """

    return - Y * np.log(Y_hat) - (1 - Y) * np.log(1 - Y_hat)

def lossFunctionCrossEntropyDerivative(Y, Y_hat):
    """
    Calculates the derivative (relative to y_hat) value of
    cross-entropy cost function at given point.
    Parameters:
        Y 		- [float: ny x m] the correct answer
        Y_hat   - [float: ny x m] the answer prediction
    """
    return - (Y / Y_hat) + (1 - Y) / (1 - Y_hat)

def costFunction(Y, Y_hat):
    """
    Calculates the total cost function for a set of examples.

    Parameters:
        Y 		- [float: ny x m] the correct answer
        Y_hat   - [float: ny x m] the answer prediction
    """

    # compute the loss for every example we have
    losses = lossFunctionCrossEntropy(Y, Y_hat)

    # summ losses to get total unnormalized cost value
    cost = np.sum(losses)

    # the total cost normalization
    cost /= losses.size

    return cost

def costFunctionDerivative(Y, Y_hat):
    """
    Calculates the total cost function derivative (relatve
    to Y_hat and at point Y_hat) for a set of given examples Y.

    Parameters:
        Y 		- [float: ny x m] the correct answer (treated
                  as const).
        Y_hat   - [float: ny x m] the answer prediction the variable.

    Returns:
        dJ(Y_hat)/dY_hat - the matrix of partial derivatives relative
                  to the corresponding Y_hat_i variable.
    """

    # compute the partial derivative for each answer y_hat
    derivatives_y_hat = lossFunctionCrossEntropyDerivative(Y, Y_hat)

    # the total cost function derivatives normalization
    derivatives_y_hat /= derivatives_y_hat.size

    return derivatives_y_hat

def backwardPropagationLogisticRegression(params, X, Y, Y_hat):
    """
    Calculates the derivatives relative to linear regression
    parameters: W1 and b1.
    Parameters:
        params 	- dictionary with relevant data
        X		- [float{nx,m}] the horizontal stack of vectors
                  (column vectors) of input shape(X) = (nx, m),
                  where m is a number of examples and nx is an
                  input vector size.
        Y 		- [float: ny x m] the correct answer (treated
                  as const).
        Y_hat   - [float: ny x m] the answer prediction the variable.
    """

    # cost function calculation for reports
    J = costFunction(Y, Y_hat)
    params["J"] = J

    # getting cached parameters from forward propagation step
    Z = params["Z"]

    m = np.shape(X)[1]

    # initial derivative dJ/dY_hat
    dY_hat = costFunctionDerivative(Y, Y_hat)

    # dJ/dZ = dJ/dA * dA/dZ, note that here the multiplication
    # is element-wise
    dZ = dY_hat * sigmoidDerivative(Z)

    # Z = WX + B
    # and according to the matrix differentiations rules:
    #	if C = AB, then:
    # 	dF/dA = (dF/dC)B'
    # where B' denotes B transposed matrix
    dW = np.dot(dZ, X.T)
    dB = np.sum(dZ, 1)

    params["dW"] = dW
    params["db"] = dB

    return (dW, dB)

def initialize(params, nx):
    """
    Initializes the parameters of logistic regression.
    Parameters:
        params 	- the dictionary with parameters
    """

    # Note that the division by nx is important
    # as long as if the number of dimensions of input space
    # grows, we need to normalize it by nx to avoid increasing
    # of scalar product absolute value.
    W = np.random.randn(1, nx) / nx
    B = np.zeros((1,1))

    params["W"] = W
    params["B"] = B

    return (W,B)

def optimizationStep(params, X, Y, learning_rate):

    # getting current model parameters
    W = params["W"]
    B = params["B"]

    # making forward pass to calculate Y_hat predictions of
    # logistic regression
    Y_hat = forwardPropagationLogisticRegression(params, X)

    # calculating the gradients of the cost function at current
    # coordinates (W,B), provided by current predictions
    (dW, dB) = backwardPropagationLogisticRegression(params, X, Y
                                                     , Y_hat)

    # gradient descent single step with given learning rate
    W = W - learning_rate * dW
    B = B - learning_rate * dB

    params["W"] = W
    params["B"] = B

    return (W,B)

def optimize(params, X, Y, learning_rate, epochs):

    W = params["W"]
    B = params["B"]

    cost_history = np.zeros((epochs,))

    for i in range(epochs):
        optimizationStep(params, X, Y, learning_rate)
        cost_history[i] = params["J"]
        print "Epoch[" + str(i) + "], cost = " + str(params["J"])

    return (W, B, cost_history)


def generateTestDataSetSquareCircle(img_w, img_h, m):
    # fixed generation parameters
    r_min = 10
    r_max = 30

    # the size of a single input vector
    nx = img_w * img_h

    # input/output arrays
    X = np.zeros((nx, m))
    Y = np.zeros((1, m))

    for i in range(m):
        # random training set generation
        img = np.zeros((img_w, img_h))
        x = None

        # figure parameters
        r = np.random.randint(r_min, r_max + 1)
        ox = np.random.randint(r, img_w - r)
        oy = np.random.randint(r, img_h - r)

        if (np.random.rand() > 0.5):
            # circle
            for x in range(ox - r, ox + r + 1):
                for y in range(oy - r, oy + r + 1):
                    if (math.hypot(x - ox, y - oy) <= r):
                        img[x, y] = 1.0
            y = 1.0
        else:
            # square
            for x in range(ox - r, ox + r + 1):
                for y in range(oy - r, oy + r + 1):
                    img[x, y] = 1.0
            y = 0.0

        X[:, i] = np.reshape(img, (nx))
        Y[:, i] = y

    return (X, Y)


#def loadLabeledData(fileName):
#    with h5py.File(fileName, "r") as f:
#        dset = f.create_dataset("mydataset", (100,), dtype='i')
#    return X, Y

def validateModel(img_w, img_h, m, params
                  , origX = None, origY = None
                  , description = None):

    print("Model validation on " + str(m) + " examples"
          + (" (" + description + ")" if description is not None
                                    else "")
          + ":")

    # generating the validation data set
    X, Y = origX, origY
    if (origX is None) or (origY is None):
        (X, Y) = generateTestDataSetSquareCircle(img_w, img_h, m)

    Y_hat = forwardPropagationLogisticRegression(params, X)

    # getting discrete predictions
    predictions = Y_hat >= 0.5

    false_pos = np.sum((predictions - Y) > 0)
    false_neg = np.sum((Y - predictions) > 0)

    print("		False positive rate: " + str(false_pos)
          + "/" + str(m) + "  ->  "
          + str(100 * float(false_pos)/float(m)) + "%")
    print("		False negative rate: " + str(false_neg)
          + "/" + str(m) + "  ->  "
          + str(100 * float(false_neg)/float(m)) + "%")
    print("		Total error rate: " + str(false_neg + false_pos)
          + "/" + str(m) + "  ->  "
          + str(100 * float(false_neg + false_pos)/float(m)) + "%")

    return (false_pos, false_neg)

def plotExamples(X, Y, img_w, img_h
                 , positive_count = 1, negative_count = 1):
    # example plot

    threshold = 0.5

    pos = 0
    neg = 0
    # positive example:
    for i in range(np.shape(X)[1]):
        if (Y[0, i] >= threshold and pos < positive_count):
            pos += 1
            plt.imshow(np.reshape(X[:, i], (img_w, img_h)))
            plt.title("The positive example [" + str(i) + "]")
            plt.show()
        elif (Y[0, i] < threshold and neg < negative_count):
            neg += 1
            plt.imshow(np.reshape(X[:, i], (img_w, img_h)))
            plt.title("The negative example [" + str(i) + "]")
            plt.show()

def mainCircleSquareTest():
    # general parameters
    img_w = 128
    img_h = 128
    learning_rate = 0.01
    m = 100
    epochs = 10000

    # acquire training data set
    (X,Y) = generateTestDataSetSquareCircle(img_w, img_h, m)
    plotExamples(X, Y, img_w, img_h);

    # initialize the hypotesis parameters
    params = {}
    initialize(params, img_w * img_h)

    # plot neuron newly initialized weight vector
    plt.imshow(np.reshape(params["W"], (img_w, img_h)))
    plt.title("The logistic regression neuron newly initialized"
              + " weight vector, \n" + str(m) + " examples, "
              + str(epochs) + " epochs.")
    plt.show()

    # optimization
    (W, B, cost_history) = optimize(params, X, Y, learning_rate
                                    , epochs)

    # model test on training set
    validateModel(img_w, img_h, m, params, X, Y, description
                            = "training set")

    # model validation on validation set
    validateModel(img_w, img_h, m, params, description
                            = "validation set")

    # plot cost history
    plt.plot(cost_history)
    plt.title("The logistic regression cross-entropy cost function"
              + " value, \n"
              + str(m) + " examples, " + str(epochs) + " epochs.")
    plt.xlabel("Epoch");
    plt.ylabel("J(epoch)");
    plt.show()

    # plot neuron weight vector
    plt.imshow(np.reshape(params["W"], (img_w, img_h)))
    plt.title("The logistic regression neuron weight vector, \n"
              + str(m) + " examples, " + str(epochs) + " epochs.")
    plt.show()

mainCircleSquareTest()


