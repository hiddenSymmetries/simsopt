# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/Users/sienahurwitz/simsopt/thirdparty/boost/src/boost"
  "/Users/sienahurwitz/simsopt/thirdparty/boost/src/boost-build"
  "/Users/sienahurwitz/simsopt/thirdparty/boost"
  "/Users/sienahurwitz/simsopt/thirdparty/boost/tmp"
  "/Users/sienahurwitz/simsopt/thirdparty/boost/src/boost-stamp"
  "/Users/sienahurwitz/simsopt/thirdparty/boost/src"
  "/Users/sienahurwitz/simsopt/thirdparty/boost/src/boost-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/sienahurwitz/simsopt/thirdparty/boost/src/boost-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/sienahurwitz/simsopt/thirdparty/boost/src/boost-stamp${cfgdir}") # cfgdir has leading slash
endif()
