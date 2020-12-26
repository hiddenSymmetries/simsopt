cmake_minimum_required(VERSION 3.5)
project(simsgeopp)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)
find_package(NumPy REQUIRED)
#find_package(OpenMP)

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
# Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

set(CMAKE_CXX_FLAGS "-O3 -march=native")

add_subdirectory(thirdparty/pybind11)
set(XTENSOR_USE_OPENMP 0)



pybind11_add_module(simsgeopp
    src/simsgeopp/python.cpp src/simsgeopp/biot_savart.cpp src/simsgeopp/biot_savart_derivative.cpp
    )
set_target_properties(simsgeopp
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)
target_include_directories(simsgeopp PRIVATE "thirdparty/xtensor/include" "thirdparty/xtensor-python/include" "thirdparty/xsimd/include" "thirdparty/xtl/include" "thirdparty/blaze" ${NUMPY_INCLUDE_DIRS} "src/simsgeopp/")
target_link_libraries(simsgeopp PUBLIC ${PYTHON_LIBRARY})


#target_include_directories(hello_world PRIVATE  "xtensor/include" "xtensor-python/include" "xsimd/include" "xtl/include" "blaze" "cppplasmaopt" ${PYBIND11_INCLUDE_DIR} ${NUMPY_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIR})
#target_link_libraries(hello_world PUBLIC ${PYTHON_LIBRARY})


#if(OpenMP_CXX_FOUND)
#    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
#    target_link_libraries(profiling_biot_savart PRIVATE OpenMP::OpenMP_CXX)
#endif()

#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

install(TARGETS simsgeopp LIBRARY DESTINATION simsgeopp)
