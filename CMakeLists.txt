cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_COMPILER_WORKS 1)
project(simsoptpp)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(Python_FIND_STRATEGY LOCATION)
find_package(Python 3 COMPONENTS Interpreter Development.Module NumPy)

message(status "Python executable is ${Python_EXECUTABLE}")
message(status "Python Development Module found value is ${Python_Development.Module_FOUND}")
message(status "Python header dirs are ${Python_INCLUDE_DIRS}")
message(status "Python library dirs are ${Python_LIBRARY_DIRS}")
message(status "Python version is ${Python_VERSION}")

# Find pybind11
execute_process(COMMAND python -m pybind11 --cmakedir
                OUTPUT_VARIABLE pybind11_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE)
message("Result of python -m pybind11 --cmakedir: ${pybind11_DIR}")
find_package(pybind11 REQUIRED PATHS pybind11_DIR)

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

# For XSIMD usage
if(DEFINED ENV{NO_XSIMD})
    set(NO_XSIMD ON)
endif()
set(_HAS_AUTO_PTR_ETC 1)
configure_file(config.h.in config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

include(CheckCXXCompilerFlag)
IF(DEFINED ENV{CI})
    if (APPLE)
        set(CMAKE_CXX_FLAGS "-O3")
    else()
        message(STATUS "CI environment detected. Set compilation flags targetting Westmere microarch.")
        set(CMAKE_CXX_FLAGS "-O3 -march=westmere")
    endif()
elseif(DEFINED ENV{CONDA_BUILD})
    message(STATUS "conda build environment detected. Let conda set compilation flags accordingly.")
    # set(CMAKE_CXX_FLAGS "-O3 -march=ivybridge -mfma -ffp-contract=fast")
else()
    # Temporary fix till clang 15 is default on macs with Apple silicon
    message(STATUS "Local build detected. Set compilation flags accordingly (march=native).")
    unset(COMPILER_SUPPORTS_MARCH_NATIVE CACHE)
    CHECK_CXX_COMPILER_FLAG(-march=native COMPILER_SUPPORTS_MARCH_NATIVE)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        set(CMAKE_CXX_FLAGS "-O3 -march=native -ffp-contract=fast")
    elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "arm64")
        set(CMAKE_CXX_FLAGS "-O3 -mcpu=apple-a14 -ffp-contract=fast")
    else()
        set(CMAKE_CXX_FLAGS "-O3 -ffp-contract=fast")
    endif()
endif()
# Ensure all code used from Eigen does not have LGPL license:
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_MPL2_ONLY")
message(STATUS "CMAKE_CXX_FLAGS are ${CMAKE_CXX_FLAGS}")

if(DEFINED ENV{CONDA_PREFIX})
    include_directories($ENV{CONDA_PREFIX}/include)
    link_directories("$ENV{CONDA_PREFIX}/lib")
    message(STATUS "Conda prefix is $ENV{CONDA_PREFIX}")
endif()

find_package(OpenMP)
find_package(Boost 1.76.0)
if(Boost_FOUND)
    message(STATUS "Boost version is ${Boost_VERSION_STRING}")
    message(STATUS "Boost include dirs are ${Boost_INCLUDE_DIRS}")
else()
    message(STATUS "Downloading and installing boost.")
    # For some external project macros
    include(ExternalProject)

    # Download boost from git and build the headers
    set(boost_DIR ${CMAKE_CURRENT_BINARY_DIR}/thirdparty/boost)
    set(boost_target boost)

    ExternalProject_Add(
        ${boost_target}
        PREFIX ${boost_DIR}
        GIT_REPOSITORY https://github.com/boostorg/boost.git
        GIT_TAG boost-1.84.0
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
        GIT_SUBMODULES tools/build tools/boost_install libs/config libs/numeric
            libs/math libs/type_traits libs/predef libs/assert libs/static_assert
            libs/throw_exception libs/core libs/serialization libs/preprocessor libs/mpl
            libs/utility libs/typeof libs/array libs/units libs/integer libs/fusion
            libs/range libs/iterator libs/concept_check libs/detail libs/function_types
            libs/lexical_cast libs/container libs/move libs/smart_ptr libs/multi_array
            libs/functional libs/function libs/type_index libs/container_hash libs/bind 
            libs/format libs/optional
        CONFIGURE_COMMAND ./bootstrap.sh --prefix=<PREFIX>
        BUILD_COMMAND ./b2 headers --prefix=${boost_DIR}
        BUILD_IN_SOURCE 1
        INSTALL_COMMAND ""
    )

    set(Boost_INCLUDE_DIRS ${boost_DIR}/src/${boost_target})
    message(STATUS "Boost include dirs are ${Boost_INCLUDE_DIRS}")
endif()

find_package(GSL)
if(GSL_FOUND)
    add_definitions(-DUSE_GSL)
    message(STATUS "GSL version is ${GSL_VERSION}")
    message(STATUS "GSL include dirs are ${GSL_INCLUDE_DIR}")
    set(GSL_TARGETS src/simsoptpp/symplectic.cpp)
    message(STATUS "GSL targets are ${GSL_TARGETS}")
    message(STATUS "GSL libraries are ${GSL_LIBRARIES}")
else()
    set(GSL_TARGETS)
endif()

set(XTENSOR_USE_OPENMP 0)
set(XTENSOR_USE_TBB 0)

pybind11_add_module(${PROJECT_NAME}
    src/simsoptpp/python.cpp 
    src/simsoptpp/python_tracing.cpp
    src/simsoptpp/python_interpolant.cpp
    src/simsoptpp/regular_grid_interpolant_3d_py.cpp
    src/simsoptpp/tracing.cpp 
    src/simsoptpp/python_boozermagneticfield.cpp
    src/simsoptpp/boozerradialinterpolant.cpp
    ${GSL_TARGETS}
)

set_target_properties(${PROJECT_NAME}
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)

target_include_directories(${PROJECT_NAME} PRIVATE "thirdparty/xtensor/include" "thirdparty/xtensor-python/include" "thirdparty/xsimd/include" "thirdparty/xtl/include" ${Python_NumPy_INCLUDE_DIRS} "src/simsoptpp/")

if(NOT Boost_FOUND)
    add_dependencies(${PROJECT_NAME} ${boost_target})
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-I${Boost_INCLUDE_DIRS}")

if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

IF (GSL_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${GSL_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${GSL_LIBRARIES})
endif()
