cmake_minimum_required(VERSION 3.2)

# external parameters
set(GENERATE_PYTHON_STANDALONE_EXECUTABLES false CACHE BOOL
    "Set to true if to generate standalone python executables.")

# standalone python executables generation
function(standalone_python_gen targetName sources_R)
    if (GENERATE_PYTHON_STANDALONE_EXECUTABLES)
        foreach (pyscript_R ${sources_R})
            get_filename_component(srcDir_R ${pyscript_R} DIRECTORY)
            get_filename_component(srcName ${pyscript_R} NAME_WE)

            if(UNIX)
                set(exec_name "${srcName}")
            endif(UNIX)
            if(WIN32)
                set(exec_name "${srcName}.exe")
            endif(WIN32)

            message(STATUS "Standalone python generation: "
                           "${pyscript_R} -> ${exec_name}")

            set(src_A "${CMAKE_CURRENT_SOURCE_DIR}/${pyscript_R}")
            set(dstDir_A "${CMAKE_CURRENT_BINARY_DIR}/${srcDir_R}")
            get_filename_component(dst_A "${dstDir_A}/${exec_name}"
                                   ABSOLUTE)

            if(NOT EXISTS ${src_A})
                message(STATUS "(!) source file does not exist:"
                               " ${src_A}")
                continue()
            endif(NOT EXISTS ${src_A})

            add_custom_command(OUTPUT "${dst_A}"
                COMMAND pyinstaller -n "${exec_name}"
                                    --workpath "${dstDir_A}"
                                    "${src_A}"
                DEPENDS "${src_A}"
                VERBATIM
            )

            set(destinations ${destinations}
                             ${dst_A})
        endforeach (pyscript_R ${sources_R})

        add_custom_target(${targetName} ALL DEPENDS ${destinations})

    else(GENERATE_PYTHON_STANDALONE_EXECUTABLES)
        message(STATUS "Binary python executables generation disabled"
                       " see GENERATE_PYTHON_STANDALONE_EXECUTABLES "
                       " cached variable for more information.")
    endif(GENERATE_PYTHON_STANDALONE_EXECUTABLES)
endfunction(standalone_python_gen targetName sources_R)

# main section

add_custom_target(WorkshopTasksRawFiles SOURCES "README.txt"
                        "LICENSE")

add_subdirectory("workshop2")
add_subdirectory("commonlearningdata")
