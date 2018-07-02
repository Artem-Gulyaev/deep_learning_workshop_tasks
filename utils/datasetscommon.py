import numpy as np
import math

def shuffleDataset(X, Y):
    """
    Returns shuffled dataset.
    PARAMETERS:
        X		- [float: nx x m] input dataset.
        Y		- [float: ny x m] output dataset.
    RETURNS:
        (shuffledX, shuffledY) - arrays of the same shape as
                X and Y but shuffled randomly together.
    """
    m = np.shape(X)[1]

    permutation = np.random.permutation(m)

    outX = X[:, permutation]
    outY = Y[:, permutation]

    return (outX, outY)

