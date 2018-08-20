cmake_minimum_required(VERSION 3.2.2)
project(devernaysubpix)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Boost COMPONENTS timer)
find_package(OpenCV REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

set(HEADERS
    curvepoint.hpp
    draw.hpp
    edge_detector.hpp
    linkmap.hpp
    neighborhoodbitmap.hpp
    )

add_executable(devernaysubpix
    main.cpp
    ${HEADERS}
    )
target_link_libraries(devernaysubpix
  ${OpenCV_LIBS}
  Boost::timer
  )

set_target_properties(devernaysubpix PROPERTIES CXX_STANDARD 11)

# Enforce return values in non-void functions
target_compile_options(devernaysubpix PUBLIC "-Werror=return-type")


# OpenCV example

add_executable(cv_example
    ${HEADERS}
    cv_example.cpp
    )
target_link_libraries(cv_example
  ${OpenCV_LIBS}
  )

set_target_properties(cv_example PROPERTIES CXX_STANDARD 11)

# Enforce return values in non-void functions
target_compile_options(cv_example PUBLIC "-Werror=return-type")

# test_devernaysubpix

add_executable(test_devernaysubpix
    ${HEADERS}
    test_devernaysubpix.cpp
    )
target_link_libraries(test_devernaysubpix
  ${OpenCV_LIBS}
  )

set_target_properties(test_devernaysubpix PROPERTIES CXX_STANDARD 11)

# Compiler
if (WIN32)
    if (MSVC)
        add_compile_options(/W4)
    endif(MSVC)
endif()

install(TARGETS devernaysubpix cv_example RUNTIME DESTINATION bin)
