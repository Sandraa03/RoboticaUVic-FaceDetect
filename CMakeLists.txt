#indicate minimum version
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

#project name
PROJECT(FaceDetect)

#find required packages (look for the package, usually at /usr/share/cmake-2.8/Modules/ or /usr/share/)
FIND_PACKAGE(OpenCV REQUIRED)

#set header directories
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIR})

#Create an executable
ADD_EXECUTABLE(${PROJECT_NAME}_Q1 opencv-facedetect_Q1.cpp)
ADD_EXECUTABLE(${PROJECT_NAME}_Q2 opencv-facedetect_Q2.cpp)

#Link with libraries
TARGET_LINK_LIBRARIES(${PROJECT_NAME}_Q1 ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(${PROJECT_NAME}_Q2 ${OpenCV_LIBS})

# Setting this prefix will be used by INSTALL commands in next CMakeLists
SET(CMAKE_INSTALL_PREFIX /usr/local)

#install
INSTALL(TARGETS ${PROJECT_NAME}_Q1 RUNTIME DESTINATION bin)
INSTALL(TARGETS ${PROJECT_NAME}_Q2 RUNTIME DESTINATION bin)