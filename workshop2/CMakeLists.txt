cmake_minimum_required(VERSION 3.2)

set(PYTHON_MAINS "logisticregressionmain.py")

add_custom_target(CatRecognizer SOURCES ${PYTHON_MAINS}
                            "README.txt")

# note the quotes around PYTHON_MAINS are necessary
standalone_python_gen(Workshop2PythonBinaries "${PYTHON_MAINS}")
