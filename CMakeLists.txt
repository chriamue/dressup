cmake_minimum_required(VERSION 2.8)

project(dressup)

find_package( openpose COMPONENTS wrapper pose REQUIRED )
find_package( Caffe REQUIRED )
find_package( OpenCV REQUIRED )

add_executable(${PROJECT_NAME} "main.cpp")

add_definitions( -DUSE_CAFFE)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)

target_link_libraries(${PROJECT_NAME} ${Caffe_LIBRARIES} ${OpenCV_LIBS} openpose::wrapper)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/scarf.png
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/skirt.png
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/tshirt.png
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/models/pose/coco/pose_deploy_linevec.prototxt
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/models/pose/coco/)

if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/models/pose/coco/pose_iter_440000.caffemodel")
else()
    message(STATUS "downloading caffemodel")
    file(DOWNLOAD "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel"
        "${CMAKE_CURRENT_BINARY_DIR}/models/pose/coco/pose_iter_440000.caffemodel")
    message(STATUS "caffemodel downloaded")
endif()
