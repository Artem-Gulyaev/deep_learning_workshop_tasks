import numpy as np
import h5py
import math
import matplotlib.pyplot as plt
import os

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
                      , trainingError[0] + trainingError[1]]) * 100
    valid = (np.array([validationError[0], validationError[1]
                      , validationError[0] + validationError[1]])
            * 100)

    trainBars = ax.bar(locations, train, bar_width, color = 'blue')
    validBars = ax.bar(locations + bar_width, valid, bar_width
                       , color = 'black')

    ax.set_title('Training and validation set performance'
                 + ' comparison\n(' + description + ')')

    for i in range(3):
        ax.text(i - bar_width * 0.5 + 0.1, train[i] + 0.005
                , "{:.2f}".format(train[i]) + "%", color = 'blue')
    for i in range(3):
        ax.text(i + bar_width * 0.5 + 0.1, valid[i] + 0.005
                , "{:.2f}".format(valid[i]) + "%", color = 'black')

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

def plotImageExamplesBinaryClassification(X, Y, img_w, img_h
                      , positive_count = 1, negative_count = 1
                      , output_threshold = 0.5
                      , show = False):
    """
    Plots and shows positive_count of positive image examples, and
    negative_count of negative image examples for images binary
    classification problem.
    PARAMETERS:
        X			- [float: img_w * img_h x m] - the input images
                    data set. Each column is a single input image.
        Y 			- [float: 1 x m] - the corresponding output data.
        img_w,
        img_h   	- [int] - the sizes of the images: width and
                    height correspondingly.
        positive_count
                    - [int >= 0], the number of positive image
                    examples to plot.
        negative_count
                    - [int >= 0], the number of negative image
                    examples to plot.
        output_threshold
                    - [float in [0.0; 1.0]] - the threshold
                    value to distinguish between 0 class and 1 class
                    of images.
        show        - [bool] - if True, then will request to show the
                    resulting plot immediately.
    """
    threshold = output_threshold

    pos = 0
    neg = 0
    # positive example:
    for i in range(np.shape(X)[1]):
        if (Y[0, i] >= threshold and pos < positive_count):
            pos += 1
            plt.imshow(np.reshape(X[:, i], (img_w, img_h)))
            plt.title("The positive example [" + str(i) + "], ("
                      + str(img_w) + "x" + str(img_h) + ")")
            # no ticks on axes
            plt.xticks([])
            plt.yticks([])
            if (show):
                plt.show()
        elif (Y[0, i] < threshold and neg < negative_count):
            neg += 1
            plt.imshow(np.reshape(X[:, i], (img_w, img_h)))
            plt.title("The negative example [" + str(i) + "], ("
                      + str(img_w) + "x" + str(img_h) + ")")
            # no ticks on axes
            plt.xticks([])
            plt.yticks([])
            if (show):
                plt.show()

def plotHistory(cost_history, title = None, show = True):
    """
    Plots the graph of cost function value J versus optimization
    step number.
    PARAMETERS:
        cost_history -
                    * [float: n] or [float n x 1], where n is a
                      number of cost values to plot,
                    * or [float n x 2],
                      [:, 0:1] is the same as for [float n x 1]
                      [:, 1] is a training set error history, or
                    * or [float n x 3],
                      [:, 0:2] is the same as for [float n x 2]
                      [:, 2] is a validation set error history.
        title       - [str] human readable description of the
                    weights. If none, default title will be used.
        show	    - [bool] if True, then the function will
                    show plot by itself.
    """

    shape = cost_history.shape;
    size  = shape[0]

    ax_cost = plt.gca()
    ax_error = None

    cost_plot = None
    train_err_plot = None
    val_err_plot = None

    ax_cost.set_ylabel("J(epoch)")

    plotted = False
    if len(shape) == 1:
        cost_plot, = ax_cost.plot(cost_history)
        plotted = True
    elif (len(shape) == 2):
        cost_plot, = ax_cost.plot(cost_history[:, 0])
        plotted = True

    if (len(shape) == 2) and (shape[1] >= 2):
        ax_error = ax_cost.twinx()
        ax_error.set_ylim(0.0, 1.0)
        train_err_plot, = ax_error.plot(cost_history[:, 1], '.'
                                        , color = 'green')
        ax_error.set_ylabel("Errors")
        plotted = True

    if (len(shape) == 2) and (shape[1] == 3):
        val_err_plot, = ax_error.plot(cost_history[:, 2], '.'
                                      , color = 'orange')
        plotted = True

    if not plotted:
        raise ArgumentError("Unknown history format: " + str(shape))

    plt.legend([cost_plot, train_err_plot, val_err_plot]
               , ["Cost function", "Training set error"
                  , "Validation set error"])

    if (title is None):
        title = ("The training history.")
    plt.title(title)
    plt.xlabel("Epoch")
    if (show):
        plt.show()

def plotSingleNeuronWeights2D(weights, img_w, img_h
                              , title = None
                              , show = True):
    """
    Plots the 2D plot of given neuron weights.
    PARAMETERS:
        weights		- [float: nx], where nx = img_w * img_h, is the
                    array of all weights of given neuron.
        img_w
        ,img_h      - [int] the shape of weights after reshaping for
                    2D plotting.
        title       - [str] human readable description of the
                    weights. If none, default title will be used.
        show	    - [bool] if True, then the function will
                    show plot by itself.
    """
    if weights is None:
        raise ArgumentError("Can not plot None.")

    # adjust array shape
    adjusted_W = weights.flatten()
    adjusted_W.resize((img_w, img_h))

    plt.imshow(adjusted_W)

    # no ticks on axes
    plt.xticks([])
    plt.yticks([])

    if (title is None):
        title = ("A neuron weights vector visualization.")
    plt.title(title)

    if (show):
        plt.show()
