cmake_minimum_required(VERSION 3.2)

add_subdirectory("imagesaugmentation")

add_custom_target(MlUtilsCommon SOURCES
        "datasetscommon.py"
        "imagedatasets.py"
        "reports.py"
        "__init__.py")
