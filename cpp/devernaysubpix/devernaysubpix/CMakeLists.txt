cmake_minimum_required(VERSION 3.2.2)
project(devernaysubpix)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Boost COMPONENTS timer)
find_package(OpenCV REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(devernaysubpix
    main.cpp
    )
target_link_libraries(devernaysubpix
  ${OpenCV_LIBS}
  Boost::timer
  )

set_target_properties(devernaysubpix PROPERTIES CXX_STANDARD 11)

# Enforce return values in non-void functions
target_compile_options(devernaysubpix PUBLIC "-Werror=return-type")

install(TARGETS devernaysubpix RUNTIME DESTINATION bin)
