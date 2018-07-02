cmake_minimum_required(VERSION 3.2)

set(PYTHON_MAINS "fullyconnectedmain.py")

add_custom_target(CatRecognizerFullyConnected SOURCES
                            ${PYTHON_MAINS})

# note the quotes around PYTHON_MAINS are necessary
standalone_python_gen(Workshop2PythonBinaries "${PYTHON_MAINS}")
