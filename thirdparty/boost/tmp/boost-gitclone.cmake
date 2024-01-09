# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

if(EXISTS "/Users/sienahurwitz/simsopt/thirdparty/boost/src/boost-stamp/boost-gitclone-lastrun.txt" AND EXISTS "/Users/sienahurwitz/simsopt/thirdparty/boost/src/boost-stamp/boost-gitinfo.txt" AND
  "/Users/sienahurwitz/simsopt/thirdparty/boost/src/boost-stamp/boost-gitclone-lastrun.txt" IS_NEWER_THAN "/Users/sienahurwitz/simsopt/thirdparty/boost/src/boost-stamp/boost-gitinfo.txt")
  message(STATUS
    "Avoiding repeated git clone, stamp file is up to date: "
    "'/Users/sienahurwitz/simsopt/thirdparty/boost/src/boost-stamp/boost-gitclone-lastrun.txt'"
  )
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "/Users/sienahurwitz/simsopt/thirdparty/boost/src/boost"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/Users/sienahurwitz/simsopt/thirdparty/boost/src/boost'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/local/bin/git"
            clone --no-checkout --depth 1 --no-single-branch --progress --config "advice.detachedHead=false" "https://github.com/boostorg/boost.git" "boost"
    WORKING_DIRECTORY "/Users/sienahurwitz/simsopt/thirdparty/boost/src"
    RESULT_VARIABLE error_code
  )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once: ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/boostorg/boost.git'")
endif()

execute_process(
  COMMAND "/usr/local/bin/git"
          checkout "boost-1.82.0" --
  WORKING_DIRECTORY "/Users/sienahurwitz/simsopt/thirdparty/boost/src/boost"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: 'boost-1.82.0'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/usr/local/bin/git" 
            submodule update --recursive --init tools/build;tools/boost_install;libs/config;libs/numeric;libs/math;libs/type_traits;libs/predef;libs/assert;libs/static_assert;libs/throw_exception;libs/core;libs/serialization;libs/preprocessor;libs/mpl;libs/utility;libs/typeof;libs/array;libs/units;libs/integer;libs/fusion;libs/range;libs/iterator;libs/concept_check;libs/detail;libs/function_types;libs/lexical_cast;libs/container;libs/move;libs/smart_ptr;libs/multi_array;libs/functional;libs/function;libs/type_index;libs/container_hash;libs/bind
    WORKING_DIRECTORY "/Users/sienahurwitz/simsopt/thirdparty/boost/src/boost"
    RESULT_VARIABLE error_code
  )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/Users/sienahurwitz/simsopt/thirdparty/boost/src/boost'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy "/Users/sienahurwitz/simsopt/thirdparty/boost/src/boost-stamp/boost-gitinfo.txt" "/Users/sienahurwitz/simsopt/thirdparty/boost/src/boost-stamp/boost-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/Users/sienahurwitz/simsopt/thirdparty/boost/src/boost-stamp/boost-gitclone-lastrun.txt'")
endif()
