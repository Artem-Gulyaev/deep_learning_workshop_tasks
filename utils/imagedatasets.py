import numpy as np
import math
import matplotlib.pyplot as plt
import os

# for images manipulations (loading and saving)
import imageio

def generateTestDataSetSquareCircle(img_w, img_h, m):
    """
    Generates artificial data set consisting of
    images of squares (y = 1) and circles (y = 1) on uniform
    background. Note: squares and circles don't cross the
    image border. Background value: 0.0, figure value: 1.0.
    PARAMETERS:
        img_w, img_h - [int] - required image size.
        m            - [int] - required examples count
    RETURNS:
        (X,Y)        - where:
            X - [float: nx x m] - the set of unrowed to
                   column vector images. Note: nx = img_w * img_h.
                   Note: unrowing goes row by row of original image.
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

def loadBinaryLabeledImagesHelper(dirName
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
            bool - True if the image is a positive example.
                   False else.
            image - numpy array of shape (img_w, img_h)
            file name - the file name of the corresponding file
    NOTE: * directory scan is non-recursive,
          * if image doesn't fit the expected W and H,
            it is ignored,
          * files without either of two prefixes are ignored.
    """

    print("Loading images dataset from following folder: "
          + dirName)

    data = []
    for filename in os.listdir(dirName):
        fits_extension = False
        for ext in extensions:
            if (filename.endswith('.' + ext)):
                fits_extension = True
                break
        if not fits_extension:
            continue

        isPos = None
        if (filename.startswith('1_')):
            isPos = True
        elif (filename.startswith('0_')):
            isPos = False

        if (isPos is not None):
            img = imageio.imread(os.path.join(dirName, filename))
            if (np.shape(img)[0] == expected_w
                    and np.shape(img)[1] == expected_h):
                print(filename + " -> " + str(np.shape(img)) + "("
                        + ("is positive example " if isPos
                                else "is negative example")
                        + ")")
                data.append((isPos, img, filename))
            else:
                print(filename + " ignored: size mismatch,"
                      + " actual shape is " + str(img.shape)
                      + " expected is (" + str(expected_h)
                      + ", " + str(expected_h) + ")")

    print("Loaded " + str(len(data)) + " images.")
    return data

def imageAlphaOnWhiteSingle(img):
    """
    Applies the image alpha channel to render it on
    white background and returns newly rendered image.
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

def imageGrayScaleSingle(img):
    """
    Applies the transformation to grayscale for given image
    and returns newly created grayscale image.
    PARAMETERS:
        img		- numpy array, representing RGB normalized image
                  , of shape (w, h, 1 || 2 || 3)
    RETURNS:
        resulting converted to grayscale image of the same
        shape as input image along geometrix axis, but with single
        color channel.
    """
    if (len(np.shape(img)) == 3):
        img = np.mean(img, 2)
    return img

def imageAutocontrastSingle(img):
    """
    Applies the autocontrast to the image provided
    and returns newly created autocontrasted image.
    PARAMETERS:
        img		- numpy array, representing grayscale normalized image
                  of shape (img_w, img_h)
    RETURNS:
        resulting autocontrasted input image of the same shape
    """
    min_val = np.amin(img[:, :])
    max_val = np.amax(img[:, :])
    img = ((img[:, :] - min_val) / (max_val - min_val + 1))
    return img

def loadBinaryLabeledImagesDataset(dirName
        , expected_w = 64, expected_h = 64
        , extensions = ("png", "jpg", "jpeg")):
    """
    Provides the autocontrasted, normalized to range [0.0; 1.0]
    grayscale images (if alpha channel was in the original image
    then white background is applied to get rid of alpha)
    from given directory as binary labeled dataset arrays.
    PARAMETERS:
        loadBinaryLabeledImagesHelper description for details.
    RETURNS:
        (X,Y)        - where
            X - [float: nx x m] - the set of unrowed to
                vertical vector images, nx = expected_w * expected_h
                m - is a number of examples.
            Y - [float: 1 x m]  - the set of correct answers
                1.0 for positive answer, 0.0 for negative.
    NOTES: * see loadBinaryLabeledImagesHelper.
    """

    img_w = expected_w
    img_h = expected_h

    data = loadBinaryLabeledImagesHelper(dirName, img_w , img_h
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

        img = imageAlphaOnWhiteSingle(img)
        img = imageGrayScaleSingle(img)
        img = imageAutocontrastSingle(img)

        X[:, i] = np.reshape(img, nx)
        Y[:, i] = label

    return (X, Y)
