CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)
FIND_PACKAGE(deal.II 9.0.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} $ENV{DEAL_II_DIR}
  )

IF (${deal.II_FOUND})
  DEAL_II_INITIALIZE_CACHED_VARIABLES()
ENDIF()

PROJECT(CFL)

INCLUDE(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++1z" COMPILER_SUPPORTS_CXX1Z)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
CHECK_CXX_COMPILER_FLAG("/std:c++latest" COMPILER_SUPPORTS_CXXLATEST)
IF(COMPILER_SUPPORTS_CXX1Z)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z")
  SET(DEAL_II_CXX_FLAGS "${DEAL_II_CXX_FLAGS} -std=c++1z")
ELSEIF(COMPILER_SUPPORTS_CXX1/)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
  SET(DEAL_II_CXX_FLAGS "${DEAL_II_CXX_FLAGS} -std=c++17")
ELSEIF(COMPILER_SUPPORTS_CXXLATEST)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++latest")
  SET(DEAL_II_CXX_FLAGS "${DEAL_II_CXX_FLAGS} /std:c++latest")
ELSE()
  MESSAGE(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
ENDIF()
INCLUDE(CMakeDependentOption)

OPTION(PVS-Analysis "Use static code analyzer PVS-Studio for applications?" OFF)
OPTION(COMPONENT_LATEX "Build LaTeX backend?" ON)
CMAKE_DEPENDENT_OPTION(COMPONENT_DEAL_II_MATRIXFREE "Build MatrixFree backend?" OFF "deal.II_FOUND" OFF)
CMAKE_DEPENDENT_OPTION(COMPONENT_DEAL_II_MESHWORKER "Build MeshWorker backend?" OFF "deal.II_FOUND" OFF)
OPTION(RUN_TESTS "Run tests after build?" OFF)
OPTION(BUILD_DOCUMENTATION "Build doxygen documentation?" OFF)

IF(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  MESSAGE(STATUS "Setting build type to 'Debug' as none was specified.")
  SET(CMAKE_BUILD_TYPE "Debug" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
               "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
ENDIF()

#setup of main target and include files
INCLUDE_DIRECTORIES(include)
FILE(GLOB SOURCES_CFL "sources/base/*.cc")

IF(COMPONENT_LATEX)
  FILE(GLOB SOURCES_LATEX "sources/latex/*.cc")
  LIST(APPEND SOURCES_CFL ${SOURCES_LATEX})
ENDIF()

IF(COMPONENT_DEAL_II_MATRIXFREE)
  FILE(GLOB SOURCES_DEAL_II_MATRIXFREE "sources/matrixfree/*.cc")
  LIST(APPEND SOURCES_CFL ${SOURCES_DEAL_II_MATRIXFREE})
ENDIF()

IF(COMPONENT_DEAL_II_MESHWORKER)
  FILE(GLOB SOURCES_DEAL_II_MESHWORKER "sources/meshworker/*.cc")
  LIST(APPEND SOURCES_CFL ${SOURCES_DEAL_II_MESHWORKER})
ENDIF()

ADD_LIBRARY(cfl ${SOURCES_CFL})

IF (${deal.II_FOUND})
  DEAL_II_SETUP_TARGET(cfl)
ENDIF()

#SET_PROPERTY(
    #    TARGET cfl
    #    PROPERTY CXX_CLANG_TIDY "clang-tidy;-warnings-as-errors=*;-checks=performance*;-header-filter=${CMAKE_SOURCE_DIR}/include/*/*.h")
#SET_PROPERTY(
#    TARGET cfl
#    PROPERTY CXX_INCLUDE_WHAT_YOU_USE)
SET_PROPERTY(
    TARGET cfl
    PROPERTY LINK_WHAT_YOU_USE)
#SET_PROPERTY(
#    TARGET cfl
#    PROPERTY CXX_CPPLINT cpplint.py)


IF(COMPONENT_LATEX)
  ADD_SUBDIRECTORY(applications/latex)
ENDIF()

IF(COMPONENT_DEAL_II_MATRIXFREE)
  ADD_SUBDIRECTORY(applications/matrixfree)
ENDIF()

IF(COMPONENT_DEAL_II_MESHWORKER)
  ADD_SUBDIRECTORY(applications/meshworker)
ENDIF()

IF(BUILD_DOCUMENTATION)
  ADD_SUBDIRECTORY(doc)
ENDIF()

ADD_SUBDIRECTORY(tests/support)

# Enable testing and descent into tests/ subdirectory:
ENABLE_TESTING()
IF (NOT ${deal.II_FOUND})
  INCLUDE(cmake/macro_add_test.cmake)
  INCLUDE(cmake/macro_pickup_tests.cmake)
ENDIF()

IF(COMPONENT_LATEX)
  ADD_SUBDIRECTORY(tests/latex)
ENDIF()

IF(COMPONENT_DEAL_II_MATRIXFREE)
  ADD_SUBDIRECTORY(tests/matrixfree)
ENDIF()

IF(COMPONENT_DEAL_II_MESHWORKER)
  ADD_SUBDIRECTORY(tests/meshworker)
ENDIF()

IF(RUN_TESTS)
  add_custom_target(all_tests ALL)
  add_custom_command(TARGET all_tests
                     COMMENT "Run tests"
                     POST_BUILD COMMAND ctest -j -V --output-on-failure
                    )
ENDIF()
