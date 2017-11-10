cmake_minimum_required(VERSION 3.2.2)
project(devernaysubpix)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

find_package(OpenCV REQUIRED)
find_package(Qt5Widgets)

include_directories(${PROJECT_SOURCE_DIR}/include)

set (CMAKE_BUILD_TYPE "DEBUG" CACHE STRING "build type?")
set (OPTIMIZE_NATIVE ON CACHE BOOL "optimize program for the compiling platform?")
set (FAST_MATH OFF CACHE BOOL "use fast math?")

add_executable(devernaysubpix
    main.cpp
    mainwindow.cpp
    edge_detector.cpp
    curvepoint.cpp
    linkmap.cpp
    )
target_link_libraries(devernaysubpix
  Qt5::Widgets
  ${OpenCV_LIBS})

set_target_properties(devernaysubpix PROPERTIES COMPILE_FLAGS "-std=gnu++17")

if(OPTIMIZE_NATIVE)
    get_target_property(CURRENT_TARGET_FLAGS devernaysubpix COMPILE_FLAGS)
    set_target_properties(devernaysubpix PROPERTIES COMPILE_FLAGS "${CURRENT_TARGET_FLAGS} -march=native")
endif()
if(FAST_MATH)
    get_target_property(CURRENT_TARGET_FLAGS devernaysubpix COMPILE_FLAGS)
    set_target_properties(devernaysubpix PROPERTIES COMPILE_FLAGS "${CURRENT_TARGET_FLAGS} -ffast-math")
endif()

install(TARGETS devernaysubpix RUNTIME DESTINATION bin)
