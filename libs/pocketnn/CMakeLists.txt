cmake_minimum_required(VERSION 3.0.0)

set(PROJECT_NAME
    pocketnnLib
)
set(LIBRARY_NAME
    pocketnn
)
set(LIBRARY_HEADERS_DIR
    include/${LIBRARY_NAME}
)
set(LIBRARY_HEADERS
    ${LIBRARY_HEADERS_DIR}/pktnn.h
    ${LIBRARY_HEADERS_DIR}/pktnn_consts.h
    ${LIBRARY_HEADERS_DIR}/pktnn_tools.h
    ${LIBRARY_HEADERS_DIR}/pktnn_mat.h
    ${LIBRARY_HEADERS_DIR}/pktnn_loader.h
    ${LIBRARY_HEADERS_DIR}/pktnn_mat3d.h
    ${LIBRARY_HEADERS_DIR}/pktnn_actv.h
    ${LIBRARY_HEADERS_DIR}/pktnn_fc.h
    ${LIBRARY_HEADERS_DIR}/pktnn_layer.h
    ${LIBRARY_HEADERS_DIR}/pktnn_loss.h
)
set(LIBRARY_SOURCE_DIR
    src
)
set(LIBRARY_SOURCE
    ${LIBRARY_SOURCE_DIR}/pktnn_tools.cpp
    ${LIBRARY_SOURCE_DIR}/pktnn_mat.cpp
    ${LIBRARY_SOURCE_DIR}/pktnn_loader.cpp
    ${LIBRARY_SOURCE_DIR}/pktnn_mat3d.cpp
    ${LIBRARY_SOURCE_DIR}/pktnn_actv.cpp
    ${LIBRARY_SOURCE_DIR}/pktnn_fc.cpp
    ${LIBRARY_SOURCE_DIR}/pktnn_layer.cpp
    ${LIBRARY_SOURCE_DIR}/pktnn_loss.cpp
)

project(${PROJECT_NAME})

add_library(${LIBRARY_NAME} STATIC
    ${LIBRARY_HEADERS}
    ${LIBRARY_SOURCE}
)

target_include_directories(${LIBRARY_NAME} PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${LIBRARY_NAME}>
    $<INSTALL_INTERFACE:include/${LIBRARY_NAME}>
)

target_include_directories(${LIBRARY_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)


# add_subdirectory(test)