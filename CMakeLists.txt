cmake_minimum_required(VERSION 3.13)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # for SonarLint

set(PROJECT_NAME
  PocketHHE 
)

set(SOURCE_DIR
    src
)

set(PROJECT_SOURCE
    ${SOURCE_DIR}/main.cpp
    ${SOURCE_DIR}/examples/pktnn_examples.cpp
    ${SOURCE_DIR}/examples/hhe_pktnn_examples.cpp
    ${SOURCE_DIR}/util/sealhelper.cpp
    ${SOURCE_DIR}/util/pastahelper.cpp
    ${SOURCE_DIR}/util/utils.cpp
    ${SOURCE_DIR}/pasta/pasta_3_plain.cpp
    ${SOURCE_DIR}/pasta/pasta_3_seal.cpp
    ${SOURCE_DIR}/pasta/SEAL_Cipher.cpp
    configs/config.cpp
    tests/ecg_tests.cpp
    tests/mnist_tests.cpp
)

set(LIBRARIES_DIR
    libs
)

set(LIBRARIES 
    pocketnn
    aes
    keccak
)

project(${PROJECT_NAME})

find_package(SEAL 4.0 REQUIRED)

# for AES code
include(CheckCCompilerFlag)
function(check_c_compiler_flag_and_add flag result)
  check_c_compiler_flag("${flag}" ${result})
  if(${result})
    add_compile_options("${flag}")
  endif()
endfunction()

if(MSVC)
  add_definitions("-DOC_ENABLE_SSE2")
  message("-- Enabling SSE2 for AES")
  add_definitions("-DOC_ENABLE_AESNI")
  message("-- Enabling AES_NI")
else()
  check_c_compiler_flag_and_add(-msse2 CC_SUPPORTS_SSE2)
  if(CC_SUPPORTS_SSE2)
    add_definitions("-DOC_ENABLE_SSE2")
    message("-- Enabling SSE2 for AES")
  endif()

  check_c_compiler_flag_and_add(-maes CC_SUPPORTS_AES)
  if(CC_SUPPORTS_AES)
    add_definitions("-DOC_ENABLE_AESNI")
    message("-- Enabling AES_NI")
  else()
    add_definitions("-DOC_ENABLE_PORTABLE_AES")
  endif()
endif()


add_executable(${PROJECT_NAME} ${PROJECT_SOURCE})

foreach(LIBRARY ${LIBRARIES})
    add_subdirectory("${LIBRARIES_DIR}/${LIBRARY}")
    message(STATUS "Libraries included: ${LIBRARIES_DIR}/${LIBRARY}")
endforeach(LIBRARY)
target_link_libraries(${PROJECT_NAME} ${LIBRARIES})
target_link_libraries(${PROJECT_NAME} SEAL::seal)