project(mrdserver)
cmake_minimum_required(VERSION 3.17)

option(BUILD_CSRC "Build C++ ISMRMRD client library" ON)

if(BUILD_CSRC)
    add_subdirectory(csrc)
endif()
