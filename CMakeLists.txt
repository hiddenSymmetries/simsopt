cmake_minimum_required(VERSION 3.5)
project(simsgeopp)
#set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

#find_package(PythonInterp 3 REQUIRED)
#find_package(PythonLibs 3 REQUIRED)
#find_package(NumPy REQUIRED)

set(Python_FIND_STRATEGY LOCATION)
find_package(Python 3 COMPONENTS Interpreter Development.Module NumPy)

find_package(OpenMP)
message(status "Python executable is ${Python_EXECUTABLE}")
message(status "Python Development Module found value is ${Python_Development.Module_FOUND}")
message(status "Python header dirs are ${Python_INCLUDE_DIRS}")
message(status "Python library dirs are ${Python_LIBRARY_DIRS}")
message(status "Python version is ${Python_VERSION}")

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

IF(DEFINED ENV{CI})
    message(STATUS "CI environment detected. Set compilation flags accordingly (target ivybridge which supports avx).")
    set(CMAKE_CXX_FLAGS "-O3 -march=ivybridge -mfma -ffp-contract=fast")
else()
    message(STATUS "Local build detected. Set compilation flags accordingly (march=native).")
    set(CMAKE_CXX_FLAGS "-O3 -march=native -mfma -ffp-contract=fast")
endif()

add_subdirectory(thirdparty/pybind11)
set(XTENSOR_USE_OPENMP 0)


pybind11_add_module(${PROJECT_NAME}
    src/simsgeopp/python.cpp src/simsgeopp/biot_savart_py.cpp src/simsgeopp/biot_savart_vjp_py.cpp
    )
set_target_properties(${PROJECT_NAME}
    PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)
target_include_directories(${PROJECT_NAME} PRIVATE "thirdparty/xtensor/include" "thirdparty/xtensor-python/include" "thirdparty/xsimd/include" "thirdparty/xtl/include" "thirdparty/eigen" ${Python_NumPy_INCLUDE_DIRS} "src/simsgeopp/")

add_executable(profiling profiling/profiling.cpp src/simsgeopp/biot_savart_c.cpp src/simsgeopp/biot_savart_vjp_c.cpp)
set_target_properties(profiling
    PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)
target_include_directories(profiling PRIVATE  "thirdparty/xtensor/include" "thirdparty/xsimd/include" "thirdparty/xtl/include" "thirdparty/eigen" "src/simsgeopp/")


if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

#install(TARGETS ${PROJECT_NAME}
#        #LIBRARY
#        DESTINATION src/${PROJECT_NAME})
#install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${PROJECT_NAME})
