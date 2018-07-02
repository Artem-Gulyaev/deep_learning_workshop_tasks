cmake_minimum_required(VERSION 3.2)

add_custom_target(IMAGE_DATASET_AUGMENTATION SOURCES
                        "config.json"
                        "generator.py")
