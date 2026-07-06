cmake_minimum_required(VERSION 3.7)
project(InterfaceGrid VERSION 1.0 LANGUAGES CXX)
set(CMAKE_VERBOSE_MAKEFILE on)

include(GNUInstallDirs)
set(CMAKE_CXX_STANDARD 14)
set(GAMB_ROOT "" CACHE PATH "Path to the gamb headers/library root")
set(DOCKINGLIB_ROOT "" CACHE PATH "Path to the DockingLib headers/library root")

set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
set(BUILD_SHARED_LIBS OFF)
set(ENABLE_STATIC ON)
set(CMAKE_EXE_LINKER_FLAGS "-static")
set(Boost_USE_STATIC_LIBS ON)
set(HDF5_USE_STATIC_LIBRARIES ON)

FIND_PACKAGE(Boost COMPONENTS program_options system filesystem REQUIRED)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")

file(GLOB SOURCE_FILES_GRID
        interface.cc cnpy.cc
        )
add_executable(interface2grid ${SOURCE_FILES_GRID})
target_include_directories(interface2grid PRIVATE ${GAMB_ROOT} ${DOCKINGLIB_ROOT} ${Boost_INCLUDE_DIRS})
if(GAMB_ROOT)
    target_link_directories(interface2grid PRIVATE ${GAMB_ROOT})
endif()
if(DOCKINGLIB_ROOT)
    target_link_directories(interface2grid PRIVATE ${DOCKINGLIB_ROOT})
endif()
target_link_libraries(interface2grid z dockingLib gamb++ boost_program_options boost_system boost_filesystem m)
