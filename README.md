# Deep Learning workshop workspace
This repo contains datasets and Machine Learning tasks/solutions
related the the Machine Learning workshop.

## Getting started
- Register on GitHup if not yet there
- Fork this repository (we will work via pull requests)
- Create your personal branch in forked repo (name is up to you)
- Install python (2.7 is OK)
- Install required python modules:
```
    # numpy to be installed (for vectorized computations)
    sudo pip install numpy

    # matplotlib to be installed (for plotting the data)
    sudo pip install matplotlib

    # h5py to be installed (for H5 files support,
    # which are designed to store nonuniform
    # data structures, like various data-sets)
    sudo pip install h5py

    # [optional] pyinstaller to be installed (for standalone
    # python scripts executables)
    # NOTE: you don't install pyinstaller, then
    #	do not set GENERATE_PYTHON_STANDALONE_EXECUTABLES
    #   CMake parameter to True.
    sudo pip install pyinstaller
```

## General workflow

1. Commit your solution to *correct branch* in *forked repo*,
   - if you need to commit new data to the dataset, use *master branch*
     to make your examples be available to everyone.
   - if you need to commit your solution for the task
     - use your personal branch
     - commit to subfolder of relevant workshop folder:
       e.g. for John Smith:
       ```
        <branch jsmith>
        PROJECT_ROOT/workshop2/jsmith/mycoolsolution1.py
        PROJECT_ROOT/workshop2/jsmith/mycoolsolution2.py
       ```
   - if you need to commit some reasonable improvement for the
     whole project, please also use *master branch*
2. Create pull request for common repository.
3. Update your commit according to comments.
4. As your pull request is approved, your changed will be added
   to the main repo.

## For convenience

You can add custom run procedures in your IDE, to make it convenient
to start relevant python script directly from IDE.
