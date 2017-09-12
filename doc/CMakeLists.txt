cmake_minimum_required(VERSION 3.2)

# Find doxygen
FIND_PACKAGE(Doxygen)
IF(NOT DOXYGEN_FOUND)
  MESSAGE(FATAL_ERROR
    "Could not find doxygen which is required for building the documentation"
    )
ENDIF()

# Input files for doxygen
SET(DOXYGEN_INPUT ${PROJECT_SOURCE_DIR}/..)
SET(DOXYGEN_STRIP_FROM_PATH ${PROJECT_SOURCE_DIR})

# Generate doxyfile and put it into build directory
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/options.dox.in
  ${CMAKE_CURRENT_BINARY_DIR}/options.dox
  @ONLY
  )

# Call doxygen in build dir
ADD_CUSTOM_COMMAND(
  OUTPUT ${CMAKE_BINARY_DIR}/doxygen.log
  COMMAND 
  ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/options.dox &>
  ${CMAKE_BINARY_DIR}/doxygen.log
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/options.dox
  COMMENT "Generating documentation."
  VERBATIM
  )

# Add target
ADD_CUSTOM_TARGET(doxygen
  DEPENDS ${CMAKE_BINARY_DIR}/doxygen.log
  )
