import numpy as np
import math
import matplotlib.pyplot as plt
import os
import sys

# for cmd line arguments parsing
import argparse

# for images manipulations (loading and saving)
import imageio

# adding tasks directory as a parent
sys.path.append(os.path.dirname(os.path.dirname(
                                    os.path.abspath(__file__) ) ) )
# to generate our artificial images dataset
from utils import imagedatasets
from utils import datasetscommon
from utils import reports

###
### This is an example of regular fully connected
### neural network consisting of several layers
### with sigmoid activation function,
### implementation, including training.
###

######################################
####  MACHINE LEARNING MODEL PART ####
######################################

def sigmoid(X):
    """
    Calculates the element-wise sigmoid of its argument:
        sigmoid(x) = 1 / (1 + exp(-x))
    PARAMETERS:
        X    	- the numpy array of arbitrary size and shape or a
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
                  to calculate derivatives.
    RETURNS:
        The by-element computed sigmoid derivative for all
        elements of X. Shape of X is kept.
    """

    sigma = sigmoid(X)

    # note, that "*" means element-wise product in numpy
    return sigmoidDerivativeOut(sigma)

def sigmoidDerivativeOut(sigma):
    """
    Fast way to calculate the sigmoid derivative at given value(s)
    of sigmoid.
    PARAMETERS:
        sigma     - numpy array or single value of values of sigmoid
                  where to calculate derivatives.
    RETURNS:
        The by-element computed sigmoid derivative for all
        elements of A. Shape of A is kept.
    """
    return sigma * (1 - sigma)

def forwardPropagationFullyConnected(params, X):
    """
    Calculates the output of the model for given input dataset X,
    and for parameters defined by params.
    PARAMETERS:
        params 	- the dictionary containing the relevant parameters,
                params["W#"] - [float{n^{l - 1}, n^{l}}] - the weights
                    matrix of the neurons on layer l = #. i-th column
                    of the matrix W# is a basis vector of the
                    i-th neuron in layer #.
                params["b#"] - [float{n^{l},1}] - the biases of the
                    neurons on layer l = #. The i-th componen of
                    b# corresponds to the bias value of the i-th
                    neuron on layer #.
                params["N"] - [int > 0] - the number of layers
                    in the model (all layers are fully connected).
                    The sizes of the layers are taken from the
                    shapes of corresponding W# matrices.
        X		- [float{nx,m}] the horizontal stack of vectors
                (column vectors) of input shape(X) = (nx, m),
                where m is a number of examples and nx is an
                input vector size.
    RETURNS:
        Y_hat   - [float{ny, m}] our hypotesis predictions for all
                m input examples of X. Where, ny is defined by the
                size of the output layer
    """

    # total number of layers with neurons
    N = params["N"]

    # input of the first layer is our global input X
    A_prev = X
    params["A" + str(0)] = X

    for l in range(1, N + 1):
        # W in this case is a matrix of shape (n^{[l - 1]}, n^{[l]})
        # which is stacked horizontally basis vectors of all neurons
        # at layer l. Size of each basis vector is n^{[l]}
        W = params["W" + str(l)]

        # B is a vector of all biases of neurons at given layer l.
        # shape(B) == (n^{[l]}, 1)
        b = params["b" + str(l)]

        # We project all our input vectors of given layer onto
        # new basis of n^{[l]} vectors defined in the input space
        # of n^{[l-1]} dimension, and adding a bias vector b.
        # A_prev is an output of the previous layer of shape
        # (n^{[l - 1]}, m).
        # Note: the bias vector will be broadcasted along its second
        #     index from 1 to m.
        # Output is intermediate result of linear part of neuron
        # transformation.
        Z = np.dot(W.T, A_prev) + b

        # Activation of current layer l. For now always use a sigmoid.
        # However the choose of activation function is up to algorithm
        # design.
        A = sigmoid(Z)

        # saving resulting layer activations
        params["A" + str(l)] = A
        # saving also intermediate data to avoid excessive
        # computations on backward propagation step
        params["Z" + str(l)] = Z

        # preparing data for the next layer
        A_prev = A

    # output of the last layer is our hypotesis output
    Y_hat = A_prev

    return Y_hat

### NOTE: the following section is totally the same as
### for the logistic regression

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

### same as logistic regression till here ^

def backwardPropagationFullyConnected(params, X, Y, Y_hat):
    """
    Calculates all partial derivatives of our cost function relative
    to all parameters of our set of fully connected layers:
    dJ/dW# and dJ/db#, and writes them into the params dictionary
    as "dW#" and "db#".
    PARAMETERS:
        params 	- dictionary with relevant data
                see forwardPropagationFullyConnected()
                for description.
        X		- [float: nx x m}] the horizontal stack of vectors
                (column vectors) of input shape(X) = (nx, m),
                where m is a number of examples and nx is an
                input vector size.
        Y 		- [float: ny x m] the correct answer (treated
                as const).
        Y_hat   - [float: ny x m] the answer prediction of our
                hypotesis.
    """

    # cost function calculation for history report
    J = costFunction(Y, Y_hat)
    params["J"] = J

    # initial derivative dJ/dY_hat of shape (1, m)
    dY_hat = costFunctionDerivative(Y, Y_hat)

    # total number of layers with neurons
    N = params["N"]

    # partial derivatives relative to all activations of
    # current layer
    dA = dY_hat;

    # for all layers with neurons (and thus with parameters):
    # from N to 1 inclusive:
    for l in range(N, 0, -1):
        # getting activations from previous and current layers
        A = params["A" + str(l)]
        A_prev = params["A" + str(l - 1)]

        # getting current layer weights
        W = params["W" + str(l)]

        # calculating the dJ/dZ for current layer.
        # NOTE:  dJ/dZ = dJ/dA * dA/dZ = dJ/dA * sigmoid'(Z)
        #     where '*' means element-wize product
        dZ = dA * sigmoidDerivativeOut(A)

        # Z = WX + b
        # and according to the matrix differentiations rules:
        #	if C = AB, then: dF/dA = (dF/dC)B'
        # where B' denotes B transposed matrix
        # So, partial derivatives dJ/dW and dJ/db will be:
        dW = np.dot(dZ, A_prev.T).T
        dA_prev = np.dot(W, dZ)
        # NOTE: keep dims keeps the dimension we summ along,
        #    this is needed to keep the correct behaviour
        #    of vector and matrix operations. So shape of db
        #    will be not (n^{[l]},) but (n^{[l]}, 1).
        db = np.sum(dZ, 1, keepdims = True)

        params["dW" + str(l)] = dW
        params["db" + str(l)] = db

        # preparing for the next step
        dA = dA_prev

def initialize(params, nx, layers_sizes):
    """
    Initializes the parameters of logistic regression.
    By zero biases vectors, and random weights vectors.
    Writes the W# and b# values to the parameters dictionary.
    For details about dictionary,
    see forwardPropagationFullyConnected() description.
    PARAMETERS:
        params 	- the dictionary with parameters to save.
        nx 		- [int] the input vector size.
        layers_sizes - [list of ints] - the list of layer sizes
                starting from l = 1 and up to l = N, layer size
                means "the number of neurons in the layer". Output
                of the last layer is interpreted as a final output
                of the model.
    """

    # Getting the number of layers.
    N = len(layers_sizes)
    params["N"] = N

    # Note that the division by size of previous layer is important
    # as long as if the number of dimensions of input space
    # grows, we need to normalize it by nx to avoid increasing
    # of scalar product absolute value.
    for l in range(1, N + 1):
        # size of current layer (in neurons)
        n = layers_sizes[l - 1]
        # size of previous layer
        n_prev = layers_sizes[l - 2] if l > 1 else nx

        # initialization
        # Note, that W is a horizonal stack of column basis vectors,
        # so its shape is (previous layer size ; current layer size).
        params["W" + str(l)] = np.random.randn(n_prev, n) / n_prev
        print("Generated W" + str(l) + " of shape "
              + str(params["W" + str(l)].shape))

        # Note, we init b# with zeros due to symmetry is already
        # broken by weights.
        params["b" + str(l)] = np.zeros((n, 1))
        print("Generated b" + str(l) + " of shape "
              + str(params["b" + str(l)].shape))

def optimizationStep(params, X, Y, learning_rate):
    """
    Implements a single gradient descent step and update the
    params dictionary values correspondingly.
    PARAMETERS:
        params - the dictionary with current parameters of the
                model and some intermediate calculations results:
                    params["W#"] : W# weights of # layer
                    params["b#"] : b# biases of # layer
                for more info see forwardPropagationFullyConnected.
        X		- [float: nx x m}] the horizontal stack of vectors
                  (column vectors) of input shape(X) = (nx, m),
                  where m is a number of examples and nx is an
                  input vector size.
        Y		- [float: ny x m}] the horizontal stack of vectors
                  (column vectors) of output labels shape(Y) = (ny, m)
                  where m is a number of examples and ny is an
                  output vector size.
        learning_rate -  [float] - the learning rate hyper parameter.
    """

    # layers count
    N = params["N"]

    # making forward pass to calculate Y_hat predictions of
    # logistic regression
    Y_hat = forwardPropagationFullyConnected(params, X)

    # calculating the gradients of the cost function at current
    # coordinates (W,B), provided by current predictions
    backwardPropagationFullyConnected(params, X, Y, Y_hat)

    # gradient descent single step with given learning rate
    # for all layers
    for l in range(1, N + 1):
        W = params["W" + str(l)]
        b = params["b" + str(l)]
        dW = params["dW" + str(l)]
        db = params["db" + str(l)]

        # parameters update
        W = W - learning_rate * dW
        b = b - learning_rate * db

        params["W" + str(l)] = W
        params["b" + str(l)] = b

def optimize(params, X, Y, learning_rate, epochs
             , old_history = None
             , evaluate_model_every = 0
             , valid_X = None, valid_Y = None):
    """
    The function makes epochs number optimization steps,
    for logistic regression model (one gradient descent step
    for one epoch), given by model
    parameters (contained in params), input and output
    data X, Y, learning rate (learning_rate) and number
    of iteration (epochs).
    PARAMETERS:
        params, X, Y - see optimizationStep description.
        epochs 		 - [int] - the total number of
                     optimization steps to make.
                     Note: for now number of steps is equal to number
                     of epochs.
        old_history  - [float: some_size x 3] a numpy array to append
                     new history values to, [:, 0] is a cost function
                     values. [:, 1] is a train set error values,
                     [:, 2] is a validation set error values.
        evaluate_model_every
                     - [int] - the model evaluation on training and
                     validation data sets will be made each
                     evaluate_model_every epochs, if the value is
                     <= zero, then no evaluation will be done.
        valid_X
        , valid_Y    - [float: nx x m] and [float: 1 x m] validation
                     data set. If None, than no evaluation of model
                     will be done within training process.
    RETURNS:
        cost_history -[float: optimization steps x 3] array
                     of history data values for iterations done.
                     see old_history description.
    """

    old_size = 0 if old_history is None else old_history.shape[0]

    # appending to the old Cost history if given
    cost_history = np.zeros((old_size + epochs, 3))
    if (old_size > 0):
        cost_history[ : old_size, :] = old_history

    # making requested optimization steps
    for i in range(epochs):
        # optimization step on our model
        optimizationStep(params, X, Y, learning_rate)

        # various reporting and logging actions

        curr_epoch = old_size + i

        cost_history[curr_epoch, 0] = params["J"]
        cost_history[curr_epoch, 1] = float('NAN')
        cost_history[curr_epoch, 2] = float('NAN')

        if ((evaluate_model_every > 0)
                and ((curr_epoch) % evaluate_model_every == 0)):
            (train_err, val_err) = estimateModel(params, X, Y
                                                 , valid_X, valid_Y)
            cost_history[curr_epoch, 1] = train_err[0] + train_err[1]
            cost_history[curr_epoch, 2] = val_err[0] + val_err[1]

        print("Epoch[" + str(curr_epoch) + "], cost = "
              + str(cost_history[curr_epoch, 0]))


    return cost_history

def validateModel(params, validationX, validationY
                  , description = None):
    """
    Validates the binary classification model with given params, on
    given dataset. Prints results to log and returns them as a tuple.
    PARAMETERS:
        params 		   - [dict] the model parameters dictionary.
        validationX
        , validationY  - [float: nx x m] and [float: 1 x m] labeled
                       data to validate the model. If either of two
                       is None, then (nan, nan) is returned.
        description    - [str] human readable comment for
                       validation logging. If None, not used.
    RETURNS:
        (false positive error rate [0.0; 1.0]
         , false negative error rate [0.0; 1.0])
    """

    if (validationX is None) or (validationY is None):
        return (float('NAN'), float('NAN'))

    m = np.shape(validationX)[1]

    print("Model validation on " + str(m) + " examples"
          + (" (" + description + ")" if description is not None
                                    else "")
          + ":")

    Y_hat = forwardPropagationFullyConnected(params, validationX)

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

###################################################
####   UTILITIES AND DATA REPRESENTATIOIN PART ####
###################################################

def getCatsDatasetFolder(cats_dir):
    """
    Provides the location of the cats dataset.
    PARAMETERS:
        cats_dir        - [str] - defines the directory to search
                        cats images to process, if None, then
                        default directory is used. Note: if relative
                        then counted relative to the current
                        working directory.
    RETURNS:
        the cats dataset absolute folder path
    """
    if (cats_dir is None):
        script_location = os.path.realpath(__file__)
        tasks_path = os.path.dirname(os.path.dirname(script_location))
        data_path = os.path.join(tasks_path, 'commonlearningdata')
        return os.path.join(data_path, 'cartooncats')
    else:
        return os.path.realpath(cats_dir);

def plotFullyConnectedNetworkScheme(layers, nx, show = False):
    """
    Plots the Fully Connected network scheme.
    PARAMETERS:
        layers     - [int:{L}] - array of layers sizes (in neurons).
        nx         - [int] - the input data vector size.
        show       - [bool] - if True, then will request to show the
                   resulting plot immediately.
    """

    # setting drawing limits
    ax = plt.gca()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    # no ticks on axes
    plt.xticks([])
    plt.yticks([])

    # styles for boxes
    layer_box_props = dict(boxstyle = 'round', facecolor = 'white'
                           , alpha = 0.5)
    input_box_props = dict(boxstyle = 'round', facecolor = 'green'
                           , alpha = 0.5)
    output_box_props = dict(boxstyle = 'round', facecolor = 'blue'
                           , alpha = 0.5)
    arrow_box_props = dict(boxstyle = "rarrow,pad=0.3"#, fc = "cyan"
                           , ec = "b", lw = 1)

    # layers + arrows + input + output
    boxes_count = 2 * (len(layers) + 2) - 1
    box_size = 1.0 / float(boxes_count)

    # font size
    fsize =6
    xshift = 1.2 * box_size / 2.0

    # input vector box
    plt.text(xshift, 0.5, "X\n" + str(nx), bbox = input_box_props
             , ha = "center", va = "center", fontsize = fsize)
    plt.text(xshift + box_size, 0.5, " ", bbox = arrow_box_props
             , ha = "center", va = "center", fontsize = fsize)

    for l in range(len(layers)):
        x = (2 * l + 2) * box_size
        plt.text(xshift + x, 0.5, "\n\nFC\n" + str(layers[l]) + "\n\n"
                 , bbox = layer_box_props, ha = "center"
                 , va = "center", fontsize = fsize)
        x = (2 * l + 3) * box_size
        plt.text(xshift + x, 0.5, " ", bbox = arrow_box_props
                 , ha = "center", va = "center", fontsize = fsize)

    x = (boxes_count - 1) * box_size
    plt.text(xshift + x, 0.5, "Y", bbox = output_box_props
             , ha = "center", va = "center", fontsize = fsize)

    plt.title("The Fully Connected network model.");

    if (show):
        plt.show()

def plotResults(params, X, Y, img_w, img_h
                , learning_rate, m_train, m_val, epochs
                , cost_history
                , trainingErrors, validationErrors
                , layers
                , W1 = None, W2 = None
                , W1_title = None, W2_title = None
                , show = True
                , description = None):
    """
    Plots the results for squares and circles recognigion
    via logistic regression.
    PARAMETERS:
        params 			- [dict] the model parameters and data
                        dictionary.
        X               - [float: nx x m] input training data.
                        nx = img_w * img_h.
        Y               - [float: ny x m] output training data.
                        ny = 1 for binary classification.
        img_w, img_h    - [int] dataset images width and height.
        learning_rate   - [float] learning rate
        m_train			- [int] number of taining examples
        m_val			- [int] number of validation examples
        epochs			- [int] total number of epochs to be done
        cost_history    - [float: current epochs done] the array
                        of history values for every epoch. See
                        optimize(...) return value description.
        trainingErrors
        , validationErrors
                        - [(false positive rate, false negative rate)]
                        in range [0.0; 1.0]
        layers          - [int: L] array of network layers sizes.
        W1, W2          - [float: n] some neuron weights numpy array
                        with single axes. Will be plot as image.
        W1_title
        , W2_title      - [str] titles for neuron weights.
        show 			- [bool] - if True plot will be showed.
        description		- [str] human-readable description of the
                        experiment.
    """
    plt.subplot(2, 2, 1)
    plotFullyConnectedNetworkScheme(layers, img_w * img_h
                                    , show = False)

    plt.subplot(4, 4, 3)
    reports.plotImageExamplesBinaryClassification(X, Y, img_w, img_h
                                       , positive_count = 1
                                       , negative_count = 0
                                       , show = False)
    plt.subplot(4, 4, 7)
    reports.plotImageExamplesBinaryClassification(X, Y, img_w, img_h
                                       , positive_count = 0
                                       , negative_count = 1
                                       , show = False)

    if W1 is not None:
        plt.subplot(4, 4, 4)
        reports.plotSingleNeuronWeights2D(W1, img_w, img_h
                          , W1_title, show = False)
    if W2 is not None:
        plt.subplot(4, 4, 8)
        reports.plotSingleNeuronWeights2D(W2, img_w, img_h
                          , W2_title, show = False)

    plt.subplot(2, 2, 4)
    reports.plotFullValidationResults(trainingErrors, validationErrors
                              , description = description
                              , show = False)
    plt.subplot(2, 2, 3)
    reports.plotHistory(cost_history
                , ("The regression cross-entropy cost function"
                  + " value, \n" + str(m_train) + " training examples"
                  + ", \n" + str(m_val) + " validation examples, "
                  + str(cost_history.shape[0]) + " of " + str(epochs)
                  + " epochs. Learning rate: " + str(learning_rate)
                  + ".")
                , show = False)

    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.subplots_adjust(hspace = 0.25)

    if (show):
        plt.show()

def generateModelStringDescriptor(layers):
    """
    Generates the string-descriptor of the model configuration.
    """
    out = ""
    for i in range(len(layers)):
        out += "FC" + str(layers[i]) + "_"
    out = out[0 : -1]
    return out

def generatePlotName(layers, data_description, m_train
                     , m_val, epoch, total_epochs):
    """
    Generates and returns the file name for the plot of
    current model results.
    """
    out = generateModelStringDescriptor(layers)
    out += "__" + data_description
    out += "__" + str(m_train) + "TE_" + str(m_val) + "VE"
    out += "__" + str(total_epochs) + "E"
    out += "_" + "cE" + str(epoch)
    return out

def generalOptimizationAndPlottingSequence(
            X, Y, valid_X, valid_Y, img_w, img_h
            , learning_rate, epochs_total
            , layers, plot_each_epoch, evaluate_model_every
            , show_plots, save_plots
            , experiment_description
            , data_descriptor_str):
    """
    General plotting and optimization sequence.
    PARAMETERS:
        experiment_description
                        - [str] - the human readable description of
                        the experiment.
        data_descriptor_str
                        - [str] - the part of plot file name
                        which is to describe the data set used.
    For other parameters description see: mainCircleSquareTest(...)
    """

    # initialize the hypotesis parameters
    params = {}
    initialize(params, img_w * img_h, layers)

    # initial values for cost function history
    cost_history = None

    # helper variables
    full_plots = epochs_total // plot_each_epoch
    remainder_plots = epochs_total % plot_each_epoch

    m_train = X.shape[1]
    m_val = valid_X.shape[1]

    for p in range(full_plots + 1):
        epochs = (plot_each_epoch if p < full_plots
                                  else remainder_plots)
        if (epochs == 0):
            break

        # continue optimization process
        cost_history = optimize(params, X, Y, learning_rate
                                , epochs, cost_history
                                , evaluate_model_every
                                , valid_X = valid_X
                                , valid_Y = valid_Y)


        # estimation of results of learning process
        (trainErrors, validationErrors) = estimateModel(params, X, Y
                                                        , valid_X
                                                        , valid_Y)

        fig = plt.figure(figsize=(19.60, 10.80), dpi = 300)
        # plot info about results and validate the model
        plotResults(params, X, Y, img_w, img_h
                    , learning_rate
                    , m_train, m_val
                    , epochs_total, cost_history
                    , trainErrors, validationErrors
                    , layers
                    , W1 = params["W1"][:, 0]
                    , W2 = params["W1"][:, 1]
                    , W1_title = "1st layer: 1 neuron weights ("
                                 + str(img_w) + "x" + str(img_h) + ")"
                    , W2_title = "1st layer: 2 neuron weights ("
                                 + str(img_w) + "x" + str(img_h) + ")"
                    , description = experiment_description
                    , show = show_plots)

        if (save_plots):
            plotName = generatePlotName(layers
                                   , data_descriptor_str
                                   , m_train, m_val
                                   , cost_history.shape[0]
                                   , epochs_total) + '.pdf'
            fig.savefig(plotName, dpi = 300)


###################
####   MAINS   ####
###################

def mainCircleSquareTest(
            m = 1000
            , img_w = 128
            , img_h = 128
            , learning_rate = 0.03
            , epochs_total = 20000
            , layers = [50, 30, 1]
            , plot_each_epoch = 1000
            , evaluate_model_every = 10
            , show_plots = False
            , save_plots = True):
    """
    Run this to fit the logistic regression model on an
    artificially generated dataset of images of squares and
    circles and verify the performance on training set and
    validation set.
    PARAMETERS:
        m               - [int] - number of examples to use
                        as training set.
        img_w, img_h    - [int] - generated image dimensions.
        learning_rate   - [float] - learning rate.
        epochs_total	- [int] - total epochs to be made.
        layers          - [int: L] - array of layer sizes in neurons
                        starting from first neuron layer (index 0).
        plot_each_epoch - [int(in epochs)] - intervals between plots.
        evaluate_model_every
                        - [int(in epochs)] - intervals between model
                        evaluations.
        show_plots 		- [bool] - if True, plots will be shown
                        within the process.
        save_plots      - [bool] - if True, plots will be saved
                        within the process.
    """

    # general parameters
    # layres sizes (in neurons)
    # NOTE: for binary classification problem, we need the last
    #    layer to have only one neuron.

    # acquire training and validation data sets
    (X,Y) = imagedatasets.generateTestDataSetSquareCircle(
                                        img_w, img_h, m)
    (valid_X
    , valid_Y) = imagedatasets.generateTestDataSetSquareCircle(
                                        img_w, img_h, m)

    m_train = m
    m_val = m

    generalOptimizationAndPlottingSequence(
            X, Y
            , valid_X, valid_Y
            , img_w, img_h
            , learning_rate, epochs_total
            , layers, plot_each_epoch, evaluate_model_every
            , show_plots, save_plots
            , experiment_description = "circle/square artificial"
                                       + " images with " + str(m_val)
                                       + " validation set size, and "
                                       + str(m_train) + " training"
                                       + " set size"
            , data_descriptor_str =  "squares-and-circles-artificial"
                                     + "(" + str(img_w) + "x"
                                     + str(img_h) + ")")

def mainCats(cats_dir = None
             , img_w = 64
             , img_h = 64
             , learning_rate = 0.003
             , epochs_total = 20000
             , layers = [100, 100, 50, 50, 30, 30, 1]
             , plot_each_epoch = 1000
             , evaluate_model_every = 10
             , show_plots = False
             , save_plots = True):
    """
    Run this to fit the logistic regression model on an
    external set of images of cats and non-cats
    and verify the performance on training set and
    validation set.
    PARAMETERS:
        cats_dir        - [str] - defines the directory to search
                        cats images to process, if None, then
                        default directory is used. Note: if relative
                        then counted relative to the current
                        working directory.
        img_w, img_h    - [int] - expected images dimensions.

        ALL OTHER PARAMETERS are identical
        to mainCircleSquareTest(...)
    """

    # general parameters
    cats_data_path = getCatsDatasetFolder(cats_dir)
    validation_set_size_relative = 0.10

    # acquire full data set
    (full_X, full_Y) = imagedatasets.loadBinaryLabeledImagesDataset(
                                        cats_data_path, img_w, img_h)

    if (full_X.size == 0):
        print "Loaded images dataset is empty, nothing to work with."
        return

    # as long as X, Y might not be really randomly shuffled
    # we need to shuffle data stored in X and Y
    (full_X, full_Y) = datasetscommon.shuffleDataset(full_X, full_Y)

    # computation of validation and trating sets size
    # for available data
    total_m = np.shape(full_X)[1]
    m_val = math.trunc(total_m * validation_set_size_relative)
    m_train = total_m - m_val

    print("Training set size: " + str(m_train))
    print("Validation set size: " + str(m_val))

    # extraction of training set from the data
    X = full_X[:, 0 : m_train - 1]
    Y = full_Y[:, 0 : m_train - 1]

    # validation set extraction:
    valid_X = full_X[:, m_train :]
    valid_Y = full_Y[:, m_train :]

    generalOptimizationAndPlottingSequence(
            X, Y
            , valid_X, valid_Y
            , img_w, img_h
            , learning_rate, epochs_total
            , layers, plot_each_epoch, evaluate_model_every
            , show_plots, save_plots
            , experiment_description = "Cats/non-cats dataset images"
                                       + " images with " + str(m_val)
                                       + " validation set size, and "
                                       + str(m_train) + " training"
                                       + " set size"
            , data_descriptor_str =  "cats-non-cats"
                                     + "(" + str(img_w) + "x"
                                     + str(img_h) + ")")


def main():
    parser = argparse.ArgumentParser(
                    description = 'Fully connected network example.')
    parser.add_argument('-a', '--artificial-dataset'
                        , dest = 'use_artificial_dataset'
                        , action = 'store_true'
                        , help = 'Use internally generated dataset'
                                 + ' instead of loading dataset'
                                 + ' from directory.');
    parser.add_argument('-d', '--cats-dir', dest = 'cats_dir'
                        , type = str, nargs = '?'
                        , help = 'Provides custom cats directory'
                                 + ' to process (relevant when'
                                 + ' not in artificial dataset'
                                 + ' mode).');
    parser.add_argument('-m', '--examples', nargs = '?'
                        , type = int, default = 1000, dest = 'm'
                        , help = 'Number of training examples'
                                 + ' (relevant only for artificial'
                                 + ' dataset generation).');
    parser.add_argument('-r', '--learning-rate', nargs = '?'
                        , type = float, default = 0.003
                        , dest = 'learning_rate'
                        , help = 'Learnign rate to use.');
    parser.add_argument('--img-w', nargs = '?'
                        , type = int, default = 128, dest = 'img_w'
                        , help = 'Artificial dataset images width'
                                 + ', or expected width of external'
                                 + ' images.');
    parser.add_argument('--img-h', nargs = '?'
                        , type = int, default = 128, dest = 'img_h'
                        , help = 'Artificial dataset images height'
                                 + ', or expected height of external'
                                 + ' images.');
    parser.add_argument('--epochs', nargs = '?'
                        , type = int, default = 10000, dest = 'epochs'
                        , help = 'Number epochs to be done.');
    parser.add_argument('-L', '--layers', nargs = '+'
                        , type = int, default = [50, 30, 1]
                        , dest = 'layers'
                        , help = 'Fully connected network layers'
                                 + ' sizes, say, \'50 30 1\'.');
    parser.add_argument('-p', '--plot-each', nargs = '?'
                        , type = int, default = 1000
                        , dest = 'plot_each_epoch'
                        , help = 'Number epochs to be passed between'
                                 + ' plotting.');
    parser.add_argument('-e', '--evaluate-each', nargs = '?'
                        , type = int, default = 10
                        , dest = 'evaluate_model_every'
                        , help = 'Number epochs to be passed between'
                                 + ' model evaluations.');
    parser.add_argument('--show', dest = 'show_plots'
                        , action = 'store_true'
                        , default = False
                        , help = 'Show plots within process.');
    parser.add_argument('--save', dest = 'save_plots'
                        , action = 'store_false'
                        , default = True
                        , help = 'Save plots within process.');

    args = parser.parse_args()

    print("Running with following args: " + str(args))
    plt.rc('font',size=4)
    if (args.use_artificial_dataset):
        # run with circles and squares
        mainCircleSquareTest(args.m
                             , args.img_w
                             , args.img_h
                             , args.learning_rate
                             , args.epochs
                             , args.layers
                             , args.plot_each_epoch
                             , args.evaluate_model_every
                             , args.show_plots
                             , args.save_plots)
    else:
        # run with cats from external files
        mainCats(args.cats_dir
                 , args.img_w
                 , args.img_h
                 , args.learning_rate
                 , args.epochs
                 , args.layers
                 , args.plot_each_epoch
                 , args.evaluate_model_every
                 , args.show_plots
                 , args.save_plots)

if __name__ == "__main__":
    main()
