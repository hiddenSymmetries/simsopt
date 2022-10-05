cmake_minimum_required(VERSION 3.5)
project(simsoptpp)
#set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(Python_FIND_STRATEGY LOCATION)
find_package(Python 3 COMPONENTS Interpreter Development.Module NumPy)

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

include(CheckCXXCompilerFlag)
IF(DEFINED ENV{CI})
    message(STATUS "CI environment detected. Set compilation flags accordingly (target ivybridge which supports avx).")
    set(CMAKE_CXX_FLAGS "-O3 -march=ivybridge -mfma -ffp-contract=fast")
elseif(DEFINED ENV{CONDA_BUILD})
    message(STATUS "conda build environment detected. Let conda set compilation flags accordingly.")
    # set(CMAKE_CXX_FLAGS "-O3 -march=ivybridge -mfma -ffp-contract=fast")
else()
    # Temporary fix till clang 15 is default on macs with Apple silicon
    message(STATUS "Local build detected. Set compilation flags accordingly (march=native).")
    unset(COMPILER_SUPPORTS_MARCH_NATIVE CACHE)
    CHECK_CXX_COMPILER_FLAG(-march=native COMPILER_SUPPORTS_MARCH_NATIVE)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
    	set(CMAKE_CXX_FLAGS "-O3 -march=native -mfma -ffp-contract=fast")
    elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "arm64")  
    	set(CMAKE_CXX_FLAGS "-O3 -mcpu=apple-a14 -mfma -ffp-contract=fast")
    endif()
endif()

find_package(OpenMP)
find_package(Boost 1.60.0)
if(Boost_FOUND)
    message(STATUS "Boost version is ${Boost_VERSION_STRING}")
    message(STATUS "Boost include dirs are ${Boost_INCLUDE_DIRS}")
else()
    message(STATUS "Downloading and installing boost.")
    # For some external project macros
    include(ExternalProject)

    # Download boost from git and build the headers
    set( boost_DIR ${CMAKE_CURRENT_BINARY_DIR}/thirdparty/boost )
    set( boost_target boost)

    ExternalProject_Add(
        ${boost_target}
        PREFIX ${boost_DIR}
        GIT_REPOSITORY https://github.com/boostorg/boost.git
        GIT_TAG boost-1.76.0
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
        GIT_SUBMODULES tools/build tools/boost_install libs/config libs/numeric
            libs/math libs/type_traits libs/predef libs/assert libs/static_assert
            libs/throw_exception libs/core libs/serialization libs/preprocessor libs/mpl
            libs/utility libs/typeof libs/array libs/units libs/integer libs/fusion
            libs/range libs/iterator libs/concept_check libs/detail libs/function_types
            libs/lexical_cast libs/container libs/move libs/smart_ptr libs/multi_array
            libs/functional libs/function libs/type_index libs/container_hash libs/bind
        CONFIGURE_COMMAND ./bootstrap.sh --prefix=<PREFIX>
        BUILD_COMMAND ./b2 headers --prefix=${boost_DIR}
        BUILD_IN_SOURCE 1
        INSTALL_COMMAND ""
    )

    set(Boost_INCLUDE_DIRS ${boost_DIR}/src/${boost_target})
    message(STATUS "Boost include dirs are ${Boost_INCLUDE_DIRS}")
endif()

add_subdirectory(thirdparty/pybind11)
add_subdirectory(thirdparty/fmt)
set(XTENSOR_USE_OPENMP 0)
set(XTENSOR_USE_TBB 0)


pybind11_add_module(${PROJECT_NAME}
    src/simsoptpp/python.cpp src/simsoptpp/python_surfaces.cpp src/simsoptpp/python_curves.cpp
    src/simsoptpp/python_magneticfield.cpp src/simsoptpp/python_tracing.cpp src/simsoptpp/python_distance.cpp
    src/simsoptpp/biot_savart_py.cpp src/simsoptpp/biot_savart_vjp_py.cpp
    src/simsoptpp/regular_grid_interpolant_3d_py.cpp
    src/simsoptpp/curve.cpp src/simsoptpp/curverzfourier.cpp src/simsoptpp/curvexyzfourier.cpp
    src/simsoptpp/surface.cpp src/simsoptpp/surfacerzfourier.cpp src/simsoptpp/surfacexyzfourier.cpp
    src/simsoptpp/dommaschk.cpp src/simsoptpp/reiman.cpp src/simsoptpp/tracing.cpp 
    src/simsoptpp/magneticfield_biotsavart.cpp src/simsoptpp/python_boozermagneticfield.cpp
    src/simsoptpp/boozerradialinterpolant.cpp
    )

set_target_properties(${PROJECT_NAME}
    PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)

target_include_directories(${PROJECT_NAME} PRIVATE "thirdparty/xtensor/include" "thirdparty/xtensor-python/include" "thirdparty/xsimd/include" "thirdparty/xtl/include" "thirdparty/eigen" ${Python_NumPy_INCLUDE_DIRS} "src/simsoptpp/")
target_link_libraries(${PROJECT_NAME} PRIVATE fmt::fmt)

if(NOT Boost_FOUND)
    add_dependencies(${PROJECT_NAME} ${boost_target})
endif()
target_include_directories(${PROJECT_NAME} PRIVATE ${Boost_INCLUDE_DIRS})
# target_link_libraries(${PROJECT_NAME} PRIVATE pybind11::module Boost::headers)

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(profiling EXCLUDE_FROM_ALL src/profiling/profiling.cpp src/simsoptpp/biot_savart_c.cpp src/simsoptpp/biot_savart_vjp_c.cpp src/simsoptpp/regular_grid_interpolant_3d_c.cpp)
set_target_properties(profiling
    PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)
target_include_directories(profiling PRIVATE  "thirdparty/xtensor/include" "thirdparty/xsimd/include" "thirdparty/xtl/include" "thirdparty/eigen" "src/simsoptpp/")
target_link_libraries(profiling PRIVATE fmt::fmt)



#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

#install(TARGETS ${PROJECT_NAME}
#        #LIBRARY
#        DESTINATION src/${PROJECT_NAME})
#install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${PROJECT_NAME})
