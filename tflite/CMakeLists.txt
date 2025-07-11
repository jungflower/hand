cmake_minimum_required(VERSION 3.16)

# Set app name
set(APP_NAME Handmap)

# Set 3rd party path
set(TFLite_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/include")
set(TFLite_LIBS "${CMAKE_SOURCE_DIR}/lib/libtensorflowlite.so")

# Set project
project(${APP_NAME})

# Source File
add_executable(${APP_NAME} src/main.cpp)

# Add source file
add_subdirectory(src)

# Find opengl libraries
find_package(OpenCV REQUIRED)

# Add include path
target_include_directories(${APP_NAME} 
    PRIVATE ${OpenCV_INCLUDE_DIRS} 
    PRIVATE ${TFLite_INCLUDE_DIRS})

# Link libraries to app.
target_link_libraries(${APP_NAME} 
    PRIVATE ${OpenCV_LIBS} 
    PRIVATE ${TFLite_LIBS}
)

file(COPY ${CMAKE_SOURCE_DIR}/models DESTINATION ${CMAKE_BINARY_DIR})
