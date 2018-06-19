import numpy as np
import h5py
import math
import matplotlib.pyplot as plt
import os

# for images manipulations (loading and saving)
import imageio

###
### This is an example of regular logistic regression,
### implementation, including automatic training
### testing data generation routine, and image
### data loading routine.
###

######################################
####  MACHINE LEARNING MODEL PART ####
######################################

def sigmoid(X):
    """
    Calculates the element-wise sigmoid of its argument:
        sigmoid(x) = 1 / (1 + exp(-x))
    PARAMETERS:
        X    	- the numpy array of arbitrary size or a
                  single value.
    RETURNS:
        The by-element computed sigmoid for all elements of X.
        Shape of X is kept.
    """
    return 1 / (1 + np.exp(-X))

def sigmoidDerivative(X):
    """
    Calculates the sigmoid derivative
            d\sigma/dx = \sigma(x)(1 - \sigma(x))
    for given set of X.
    PARAMETERS:
        X 		- numpy array or single value of point(s) where
                  to calculate the derivative.
    RETURNS:
        The by-element computed sigmoid derivative for all
        elements of X. Shape of X is kept.
    """

    sigma = sigmoid(X)

    # note, that "*" means element-wise product in numpy
    return sigma * (1 - sigma)

def forwardPropagationLogisticRegression(params, X):
    """
    PARAMETERS:
        params 	- the dictionary containing the relevant parameters,
                  params["W"] - the weight matrix of the neuron,
                        [float{1,nx}].
                  params["B"] - the bias of the neuron,
                        [float{1,1}].
        X		- [float{nx,m}] the horizontal stack of vectors
                  (column vectors) of input shape(X) = (nx, m),
                  where m is a number of examples and nx is an
                  input vector size.
    RETURNS:
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
    Z = np.dot(W, X) + B
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
    PARAMETERS:
        Y 		- [float: ny x m] the correct answer
        Y_hat   - [float: ny x m] the answer prediction
    RETURNS:
        the calculated element-wise cross-entropy function for
        every pair (y, y_hat) from (Y, Y_hat). Shape is preserved.
    """

    return - Y * np.log(Y_hat) - (1 - Y) * np.log(1 - Y_hat)

def lossFunctionCrossEntropyDerivative(Y, Y_hat):
    """
    Calculates the derivative (relative to y_hat) value of
    cross-entropy cost function at given point.
    PARAMETERS:
        Y 		- [float: ny x m] the correct answer
        Y_hat   - [float: ny x m] the answer prediction
    RETURNS:
        the calculated element-wise cross-entropy function derivative
        for every pair (y, y_hat) from (Y, Y_hat). Shape is preserved.
    """
    return - (Y / Y_hat) + (1 - Y) / (1 - Y_hat)

def costFunction(Y, Y_hat):
    """
    Calculates the total cost function for a set of examples.

    PARAMETERS:
        Y 		- [float: ny x m] the correct answer
        Y_hat   - [float: ny x m] the answer prediction
    RETURNS:
        single real value of the cost function corresponding
        to our predictions Y_hat and correct answers Y.
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

    PARAMETERS:
        Y 		- [float: ny x m] the correct answer (treated
                  as const).
        Y_hat   - [float: ny x m] the answer prediction the variable.

    RETURNS:
        dJ(Y_hat)/dY_hat - [float: ny x m] the matrix of partial
                  derivatives relative to the corresponding
                  Y_hat_i variable.
    NOTE:
        in logistic regression ny == 1
    """

    # compute the partial derivative for each answer y_hat
    derivatives_y_hat = lossFunctionCrossEntropyDerivative(Y, Y_hat)

    # the total cost function derivatives normalization
    # cause the J = (1/m) * summ_i(L_i)
    derivatives_y_hat /= derivatives_y_hat.size

    return derivatives_y_hat

def backwardPropagationLogisticRegression(params, X, Y, Y_hat):
    """
    Calculates the derivatives relative to linear regression
    parameters: W1 and b1.
    PARAMETERS:
        params 	- dictionary with relevant data
        X		- [float: nx x m}] the horizontal stack of vectors
                  (column vectors) of input shape(X) = (nx, m),
                  where m is a number of examples and nx is an
                  input vector size.
        Y 		- [float: ny x m] the correct answer (treated
                  as const).
        Y_hat   - [float: ny x m] the answer prediction the variable.
    RETURNS:
        (dW, dB) - the partial derivatives dJ/dW and dJ/dB.
                of W and B shape correspondingly.
    NOTE: for logistic regression ny == 1
    """

    # cost function calculation for history report
    J = costFunction(Y, Y_hat)
    params["J"] = J

    # getting cached parameters from forward propagation step
    Z = params["Z"]

    m = np.shape(X)[1]

    # initial derivative dJ/dY_hat of shape (1, m)
    dY_hat = costFunctionDerivative(Y, Y_hat)

    # dJ/dZ = dJ/dA * dA/dZ = dJ/dA * f_activation'(Z),
    # where f_activation' means the activation function derivative.
    # NOTE: that here the multiplication is element-wise
    dZ = dY_hat * sigmoidDerivative(Z)

    # Z = WX + B
    # and according to the matrix differentiations rules:
    #	if C = AB, then: dF/dA = (dF/dC)B'
    # where B' denotes B transposed matrix
    # So, partial derivatives dJ/dW and dJ/dB will be:
    dW = np.dot(dZ, X.T)
    dB = np.sum(dZ, 1)

    params["dW"] = dW
    params["db"] = dB

    return (dW, dB)

def initialize(params, nx):
    """
    Initializes the parameters of logistic regression.
    PARAMETERS:
        params 	- the dictionary with parameters to save.
        nx 		- [int] the input vector size.
    RETURNS:
        (W,B) - the initialized parameters for logistic regression
            of shapes: W: (1, nx); B: (1,1)
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
    """
    Implements a single gradient descent step.
    PARAMETERS:
        params - the dictionary with current parameters of the
                model and some intermediate calculations results:
                    params["W"] : W
                    params["B"] : B
        X		- [float: nx x m}] the horizontal stack of vectors
                  (column vectors) of input shape(X) = (nx, m),
                  where m is a number of examples and nx is an
                  input vector size.
        Y		- [float: ny x m}] the horizontal stack of vectors
                  (column vectors) of output labels shape(Y) = (ny, m)
                  where m is a number of examples and ny is an
                  output vector size.
                  NOTE: for logistic regression ny == 1
        learning_rate -  [float] - the learning rate hyper parameter.
    RETURNS:
        (W,B) 	- new values for W and B
    """

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
    """
    The function makes epochs number optimization steps,
    for logistic regression model, given by model
    parameters (contained in params), input and output
    data X, Y, learning rate (learning_rate) and number
    of iteration (epochs).
    PARAMETERS:
        params, X, Y - see optimizationStep description.
        epochs 		- [int] - the total number of
                optimization steps to make.
    RETURNS:
        (W, B, cost_history), where:
        W			- [float: 1 x nx] logistic regression
                trained model weights.
        B			- [float: 1 x 1] logistic regression
                trained model bias.
        cost_history -[float: optimization steps x 1] array
                of cost function values for iterations done.
    """

    W = params["W"]
    B = params["B"]

    cost_history = np.zeros((epochs,))

    for i in range(epochs):
        optimizationStep(params, X, Y, learning_rate)
        cost_history[i] = params["J"]
        print "Epoch[" + str(i) + "], cost = " + str(params["J"])

    return (W, B, cost_history)

def validateModel(params, validationX, validationY
                  , description = None):
    """
    Validates the logistic regression with given params, on
    given dataset. Prints results to log and returns them as a tuple.
    PARAMETERS:
        params 		   - [dict] the model parameters dictionary.
        validationX
        , validationY  - [float: nx x m] and [float: 1 x m] labeled
                       data to validate the model.
        description    - [str] human readable comment for
                       validation logging.
    RETURNS:
        (false positive error rate [0.0; 1.0]
         , false negative error rate [0.0; 1.0])
    """

    m = np.shape(validationX)[1]

    print("Model validation on " + str(m) + " examples"
          + (" (" + description + ")" if description is not None
                                    else "")
          + ":")

    Y_hat = forwardPropagationLogisticRegression(params, validationX)

    # getting discrete predictions
    predictions = Y_hat >= 0.5

    false_pos = np.sum((predictions - validationY) > 0)
    false_neg = np.sum((validationY - predictions) > 0)

    print("		False positive rate: " + str(false_pos)
          + "/" + str(m) + "  ->  "
          + str(100 * float(false_pos)/float(m)) + "%")
    print("		False negative rate: " + str(false_neg)
          + "/" + str(m) + "  ->  "
          + str(100 * float(false_neg)/float(m)) + "%")
    print("		Total error rate: " + str(false_neg + false_pos)
          + "/" + str(m) + "  ->  "
          + str(100 * float(false_neg + false_pos)/float(m)) + "%")

    return (float(false_pos) / float(m), float(false_neg) / float(m))


def estimateModel(params, trainX, trainY, validX, validY):
    """
    Calculates the training and validation datasets model performance.
    If validation dataset is not provided, then squares and circles
    validation dataset is generated on the fly.
    PARAMETERS:
        params 			- [dict] the model parameters and data
                        dictionary.
        trainX, trainY  - [float: nx x m] and [float: 1 x m] data
                        on which we have trained the model.
        validX, validY  - [float: nx x m] and [float: 1 x m] labeled
                        data to validate the model.
    RETURNS:
        (validateModel() for training set
         , validateModel() for validation set)
    NOTE:
        see validateModel() description
    """

    trainingError = validateModel(params, trainX, trainY
                                  , "training dataset")
    validationError = validateModel(params, validX, validY
                                    , "validation dataset")
    return (trainingError, validationError)

#################################
####   DATA PROVIDERS PART   ####
#################################

def generateTestDataSetSquareCircle(img_w, img_h, m):
    """
    Generates artificial data set consisting of
    images of squares (y = 1) and circles (y = 1).
    PARAMETERS:
        img_w, img_h - [int] - required image size.
        m            - [int] - required examples count
    RETURNS:
        (X,Y)        - where:
            X - [float: nx x m] - the set of unrowed to
                   column vector images. Note: nx = img_w * img_h
            Y - [float: 1 x m]  - the set of correct answers.
    """

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

def loadLabeledCatsImagesHelper(dirName
        , expected_w = 64, expected_h = 64
        , extensions = ("png", "jpg", "jpeg")):
    """
    Loads data set consisting of labeled images from
    given directory with given extensions. Interprets
    "1_" name prefix as Truth label, and "0_" as False.
    PARAMETERS:
        dirName - [str] - the path to the directory to scan.
    RETURNS:
        list of tuples (bool, image), where
            bool - True if the image is a cat picture
                   False else.
            image - numpy array
            file name - the file name of the corresponding file
    NOTE: * directory scan is non-recursive,
          * if image doesn't fit the expected W and H,
            it is ignored,
          * files without either of two prefixes are ignored.
    """

    print("Loading images dataset from following folder: "
          + dirName)

    data = []
    expected_w = 64
    expected_h = 64
    for filename in os.listdir(dirName):
        fits_extension = False
        for ext in extensions:
            if (filename.endswith('.' + ext)):
                fits_extension = True
                break
        if not fits_extension:
            continue

        isCat = None
        if (filename.startswith('1_')):
            isCat = True
        elif (filename.startswith('0_')):
            isCat = False

        if (isCat is not None):
            img = imageio.imread(os.path.join(dirName, filename))
            if (np.shape(img)[0] == expected_w
                    and np.shape(img)[1] == expected_h):
                print(filename + " -> " + str(np.shape(img)) + "("
                        + ("is a cat" if isCat else "is not a cat")
                        + ")")
                data.append((isCat, img, filename))

    print("Loaded " + str(len(data)) + " images.")
    return data

def imageAlphaOnWhite(img):
    """
    Applies the image alpha channel to render it on
    white background.
    PARAMETERS:
        img		- numpy array, representing RGBA normalized image
                  , of shape (w, h, ...)
    RETURNS:
        resulting rendered image of the same shape as input image
        along geometric coordinates, but without alpha channel values.
    """
    img_w = np.shape(img)[0]
    img_h = np.shape(img)[1]

    if (len(np.shape(img)) == 3):
        if (np.shape(img)[2] == 4):
            white = np.ones((img_w, img_h, 1))

            alpha = np.reshape(img[:, :, 3], (img_w, img_h, 1))
            img[:, :, 0:2] = (img[:, :, 0:2] * alpha
                              + white * (1 - alpha))
            img = np.delete(img, 3, 2)
    return img

def imageGrayScale(img):
    """
    Applies the transformation to grayscale for given image.
    PARAMETERS:
        img		- numpy array, representing RGB normalized image
    RETURNS:
        resulting converted to grayscale image of the same
        shape as input image along geometrix axis, but with single
        color channel.
    """
    if (len(np.shape(img)) == 3):
        img = np.mean(img, 2)
    return img

def imageAutocontrast(img):
    """
    Applies the autocontrast to the image provided
    PARAMETERS:
        img		- numpy array, representing grayscale normalized image
                  of shape (img_w, img_h)
    RETURNS:
        resulting autocontrasted input image of the same shape
    """
    min_val = np.amin(img[:, :])
    max_val = np.amax(img[:, :])
    img = ((img[:, :] - min_val) / (max_val - min_val))
    return img

def loadLabeledCatsData(dirName
        , expected_w = 64, expected_h = 64
        , extensions = ("png", "jpg", "jpeg")):
    """
    Provides the autocontrasted, normalized to range [0.0; 1.0]
    grayscale images (if alpha channel was in the original image
    then white background is applied to get rid of alpha)
    from given directory as labeled dataset arrays.
    PARAMETERS:
        loadLabeledCatsImagesHelper description for details.
    RETURNS:
        (X,Y)        - where
            X - [float: nx x m] - the set of unrowed to
                vertical vector images, nx = expected_w * expected_h
                m - is a number of examples.
            Y - [float: 1 x m]  - the set of correct answers.
    NOTES: * see loadLabeledCatsImagesHelper,
    """

    img_w = expected_w
    img_h = expected_h

    data = loadLabeledCatsImagesHelper(dirName, img_w , img_h
                                       , extensions)

    m = len(data)
    nx = img_w * img_h

    # input/output arrays
    X = np.zeros((nx, m))
    Y = np.zeros((1, m))

    for i in range(m):
        label = data[i][0]
        img = data[i][1] / 255.0
        filename = data[i][2]

        img = imageAlphaOnWhite(img)
        img = imageGrayScale(img)
        img = imageAutocontrast(img)

        X[:, i] = np.reshape(img, nx)
        Y[:, i] = label

    return (X, Y)

def shuffleDataset(X, Y):
    """
    Shuffles the given dataset and returns shuffled dataset.
    PARAMETERS:
        X		- [float: nx x m] input dataset.
        Y		- [float: ny x m] output dataset.
    RETURNS:
        (shuffledX, shuffledY) - arrays of the same shape as
                X and Y but shuffled randomly.
    """
    m = np.shape(X)[1]

    permutation = np.random.permutation(m)

    outX = X[:, permutation]
    outY = Y[:, permutation]

    return (outX, outY)

###################################################
####   UTILITIES AND DATA REPRESENTATIOIN PART ####
###################################################

def plotFullValidationResults(trainingError, validationError
                              , description = None, show = True):
    """
    Plots the model estimation results for training and validation
    datasets.
    PARAMETERS:
        trainingError  		- [(false pos rate, false negative rate)]
                            values in range [0.0; 1.0].
        validationError  	- [(false pos rate, false negative rate)]
                            values in range [0.0; 1.0].
        description		  	- [str] human readable description of the
                            experiment.
        show				- [bool] if True, then the function will
                            show plot by itself.
    """

    ax = plt.gca()

    # plotting params
    N = 3
    bar_width = 0.35

    locations = np.arange(N)
    train = np.array([trainingError[0], trainingError[1]
                      , trainingError[0] + trainingError[1]])
    valid = np.array([validationError[0], validationError[1]
                      , validationError[0] + validationError[1]])

    trainBars = ax.bar(locations, train, bar_width, color = 'blue')
    validBars = ax.bar(locations + bar_width, valid, bar_width
                       , color = 'black')

    ax.set_title('Training and validation set performance'
                 + ' comparison\n(' + description + ')')

    for i in range(3):
        ax.text(i - bar_width * 0.5 + 0.1, train[i] + 0.005
                , str(train[i]) + "%", color = 'blue')
    for i in range(3):
        ax.text(i + bar_width * 0.5 + 0.1, valid[i] + 0.005
                , str(valid[i]) + "%", color = 'black')

    ax.set_ylabel('Relative error, [%]')
    x_labels = ["False positive", "False negative", "Total"]
    ax.set_xticks(locations + bar_width)
    labels = ax.set_xticklabels(x_labels)
    ax.yaxis.grid(color = 'gray', linestyle = 'dashed'
                  , linewidth = 0.5)
    plt.legend((trainBars[0], validBars[0]), ('Training set'
                                              , 'Validation set'))
    plt.setp(labels, rotation = 0)

    if (show):
        plt.show()

def plotExamples(X, Y, img_w, img_h
                 , positive_count = 1, negative_count = 1):
    """
    Plots positive_count of positive image examples, and
    negative_count of negative image examples.
    """
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

def plotHistory(cost_history, m, epochs, show = True):
    """
    Plots the graph of cost function value J versus optimization
    step number.
    """

    plt.plot(cost_history)
    #plt.grid(color = 'gray', linestyle = 'dashed', linewidth = 0.5)
    plt.title("The logistic regression cross-entropy cost function"
              + " value, \n"
              + str(m) + " examples, " + str(epochs) + " epochs.")
    #plt.xlim(None, epochs)
    plt.xlabel("Epoch");
    plt.ylabel("J(epoch)");
    if (show):
        plt.show()

def plotNeuronWeights(params, img_w, img_h, title
                      , show = True):
    """
    Plots the 2D representation of logistic regression
    neuron weights.
    """

    plt.imshow(np.reshape(params["W"], (img_w, img_h)))
    if (title is None):
        title = ("The logistic regression neuron weight vector"
                 + " visualization.")
    plt.title(title)
    if (show):
        plt.show()

def getCatsDatasetFolder():
    """
    Provides the location of the cats dataset.
    RETURNS:
        the cats dataset folder path
    """
    script_location = os.path.realpath(__file__)
    tasks_path = os.path.dirname(os.path.dirname(script_location))
    common_data_path = os.path.join(tasks_path, 'commonlearningdata')
    cats_data_path = os.path.join(common_data_path, 'cartooncats')
    return cats_data_path

def plotLogisticRegressionResults(params, W_initial, img_w, img_h
                                  , learning_rate, m, epochs
                                  , cost_history
                                  , trainingErrors, validationErrors
                                  , separate_plots = False
                                  , description = None):
    """
    Plots the results for squares and circles recognigion
    via logistic regression.
    PARAMETERS:
        params 			- [dict] the model parameters and data
                        dictionary.
        W_initial		- [float: 1 x nx] initial weights of
                        the model.
        img_w, img_h    - [int] dataset images width and height.
        learning_rate   - [float] learning rate
        m				- [int] number of examples
        epochs			- [int] number of epochs
        cost_history    - [float: epochs] the array of cost function
                        values for every epoch.
        trainingErrors
        , validationErrors
                        - [(false positive rate, false negative rate)]
                        in range [0.0; 1.0]
        separate_plots  - [bool] - if true then all plots are done
                        independently on separate sheets.
        description		- [str] human-readable description of the
                        experiment.
    """
    # plots further
    if separate_plots == False:
        fig = plt.figure(1)
        plt.subplot(2, 2, 1)
    plotNeuronWeights({"W": W_initial}, img_w, img_h
                      , ("The logistic regression neuron newly"
                         + " initialized weight vector." )
                      , show = separate_plots)

    if separate_plots == False:
        plt.subplot(2, 2, 4)
    plotFullValidationResults(trainingErrors, validationErrors
                              , description = description
                              , show = separate_plots)
    if separate_plots == False:
        plt.subplot(2, 2, 3)
    plotHistory(cost_history, m, epochs, show = separate_plots)

    if separate_plots == False:
        plt.subplot(2, 2, 2)
    plotNeuronWeights(params, img_w, img_h
                      , ("The logistic regression neuron weight"
                         + " vector, \n" + str(m) + " examples, "
                         + str(epochs) + " epochs.")
                      , show = separate_plots)
    if separate_plots == False:
        plt.subplots_adjust(hspace = 0.25)
        plt.show()


###################
####   MAINS   ####
###################

def mainCircleSquareTest(separate_plots = False):
    """
    Run this to fit the logistic regression model on an
    artificially generated dataset of images of squares and
    circles and verify the performance on training set and
    validation set.
    PARAMETERS:
        separate_plots  - [bool] - if true then all plots are done
                        independently on separate sheets.
    """

    # general parameters
    img_w = 128
    img_h = 128
    learning_rate = 0.01
    m = 50
    epochs = 1000

    # acquire training data set
    (X,Y) = generateTestDataSetSquareCircle(img_w, img_h, m)
    plotExamples(X, Y, img_w, img_h);

    # initialize the hypotesis parameters
    params = {}
    initialize(params, img_w * img_h)
    W_initial = params["W"]

    # optimization process
    (W, B, cost_history) = optimize(params, X, Y, learning_rate
                                    , epochs)

    # validation set generation:
    (valid_X, valid_Y) = generateTestDataSetSquareCircle(img_w, img_h
                                                       , m)
    # estimation of results of learning process
    (trainErrors, validationErrors) = estimateModel(params, X, Y
                                                  , valid_X, valid_Y)

    # plot info about results and validate the model
    plotLogisticRegressionResults(params, W_initial, img_w, img_h
                                  , learning_rate, m, epochs
                                  , cost_history
                                  , trainErrors, validationErrors
                                  , separate_plots
                                  , "circle/square artificial images"
                                    + " with " + str(m)
                                    + " validation set size")

def mainCats(separate_plots = False):
    # general parameters
    cats_data_path = getCatsDatasetFolder()
    img_w = 64
    img_h = 64
    learning_rate = 0.001
    epochs = 10000
    validation_set_size_relative = 0.10

    # acquire training data set
    (full_X, full_Y) = loadLabeledCatsData(cats_data_path
                                           , img_w, img_h)
    # as long as X, Y might not be really randomly shuffled
    # we need to shuffle data stored in X and Y
    (full_X, full_Y) = shuffleDataset(full_X, full_Y)

    # computation of validation and trating sets size
    # for available data
    total_m = np.shape(full_X)[1]
    validation_m = math.trunc(total_m * validation_set_size_relative)
    m = total_m - validation_m

    print("Training set size: " + str(m))
    print("Validation set size: " + str(validation_m))
    plotExamples(full_X, full_Y, img_w, img_h, positive_count = 2
                 , negative_count = 2)

    # extraction of training set from the data
    X = full_X[:, 0 : m - 1]
    Y = full_Y[:, 0 : m - 1]

    # initialize hypotesis parameters
    params = {}
    initialize(params, img_w * img_h)
    W_initial = params["W"]

    # optimization process
    (W, B, cost_history) = optimize(params, X, Y, learning_rate
                                    , epochs)

    # validation set extraction:
    valid_X = full_X[:, m :]
    valid_Y = full_Y[:, m :]

    # estimation of results of learning process
    (trainErrors, validationErrors) = estimateModel(params, X, Y
                                                    , valid_X
                                                    , valid_Y)

    # plot info about results and validate the model
    plotLogisticRegressionResults(params, W_initial, img_w, img_h
                                  , learning_rate, m, epochs
                                  , cost_history
                                  , trainErrors, validationErrors
                                  , separate_plots
                                  , "Cats/non-cats dataset images"
                                    + " with " + str(validation_m)
                                    + " validation set size")

# to run with circles and squares
#mainCircleSquareTest(separate_plots = False)

# to run with cats
mainCats(separate_plots = False)
