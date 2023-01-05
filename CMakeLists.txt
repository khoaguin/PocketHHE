cmake_minimum_required(VERSION 3.13)

set(PROJECT_NAME
    MyPocketNN
)
set(SOURCE_DIR
    src
)
set(PROJECT_SOURCE
    ${SOURCE_DIR}/main.cpp
    ${SOURCE_DIR}/examples/pktnn_examples.cpp
    ${SOURCE_DIR}/examples/hhe_pktnn_examples.cpp
    ${SOURCE_DIR}/util/sealhelper.cpp
    ${SOURCE_DIR}/util/utils.cpp
    configs/config.cpp
)

set(LIBRARIES_DIR
    libs
)
set(LIBRARIES 
    pocketnn
)

project(${PROJECT_NAME})

find_package(SEAL 4.0 REQUIRED)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCE})

foreach(LIBRARY ${LIBRARIES})
    add_subdirectory("${LIBRARIES_DIR}/${LIBRARY}")
    message(STATUS "Libraries included: ${LIBRARIES_DIR}/${LIBRARY}")
endforeach(LIBRARY)
target_link_libraries(${PROJECT_NAME} ${LIBRARIES})
target_link_libraries(${PROJECT_NAME} SEAL::seal)