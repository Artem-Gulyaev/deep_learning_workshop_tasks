Tasks from workshop #2:
    1) Implement logistic regression forward pass algorithm

       INPUT:

         var :               :  values    -
        name :  type(shape)  :  range     - description
             :               :            -

           X : float(n_x, 1) : [0.0; 1.0] - the single normalized
                input vector of constant size n_x. Generally
                X can be a unrolled image, sound pressure values
                vector, radioactivity intensity value - anything.

       OUTPUT:
           Y_hat : float(1, 1) : [0.0; 1.0] - scalar value in
                normalized range.

       ACTIVATION FUNCTION: sigmoid

       GENERAL CONDITIONS:
           *) use only of numpy matrix and vector operations is
              recommended,
           *) watch the shapes of the vectors and matrices all the
              time, usually it is the fastest way to debug the code
              you write,
           *) try to avoid straight-forward iteration operations
              via for/while cycles, use vectorized forms of
              operations.

    2) Implement cross-entropy loss function to estimate how good
       our prediction is at given example:

       INPUT:
           Y : float(1,1) : [0.0; 1.0]     - the correct answer for
               given example.
           Y_hat : float(1,1) : [0.0; 1.0] - logistic regression
               prediction for given example.

       OUTPUT:
           L : float(1,1) : [0.0; +inf]    - the loss at given example
               given example.

    2) Commit to the "<THIS_REPO>/commonlearningdata/cartooncats"
       folder:
           *) 10 cats images ( with "1_" prefix)
           *) 10 non-cats images ( with "0_" prefix)

       REQUIREMENTS:
           *) images are of size 64x64
           *) hand-drawn by your own (style qality doesn't matter)
           *) grayscale / monochrome

       OPTIONAL:
           *) if you draw in vector originally, feel free to
              to upload vector originals into:
              "<THIS_REPO>/commonlearningdata/cartooncats/vectorized"

    3) [Optional] Implement test routine for your forward pass algorithm

       GENERAL CONDITIONS:
           *) check shapes of numpy arrays involved.
           *) check on corner cases (say, X = 0 vector).


    4) [Optional] replace the X, Y_hat and Y single column
       matrices from tasks (1 & 2) by matrices with m >= 1 columns
       and adapt your logistic regression forward pass function and
       loss calculations to make both work with multicolumn
       (many examples within the same operation) data.

       Other details are the totally similar with (1 & 2) tasks.

       GENERAL CONDITIONS:
            *) prefer vectorized calculations over straight-forward
               iteration over arrays.
