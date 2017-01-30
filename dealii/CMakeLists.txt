CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)
FIND_PACKAGE(deal.II 8.4 REQUIRED
  HINTS
  ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(cfldealii)
SET(CFL_VERSION 0.1.0)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/..)

FILE(GLOB sources *.cc)
GET_FILENAME_COMPONENT(prefix ${CMAKE_CURRENT_SOURCE_DIR} NAME)

INCLUDE(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++1y" COMPILER_SUPPORTS_CXX1Y)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
IF(COMPILER_SUPPORTS_CXX14)
  SET(DEAL_II_CXX_FLAGS "${DEAL_II_CXX_FLAGS} -std=c++14")
ELSEIF(COMPILER_SUPPORTS_CXX1Y)
  SET(DEAL_II_CXX_FLAGS "${DEAL_II_CXX_FLAGS} -std=c++1y")
ELSE()
  MESSAGE(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
ENDIF()

OPTION(PVS-Analysis "Use static code analyzer PVS-Studio?" OFF)

OPTION(DEBUG-OUTPUT "Enable verbose debug output?" OFF)
IF (MG)
  ADD_DEFINITIONS(-DDEBUG_OUTPUT)
ENDIF()

IF(PVS-Analysis)
  INCLUDE(../PVS-Studio.cmake)
ENDIF()
FOREACH(ccfile ${sources})
  GET_FILENAME_COMPONENT(file ${ccfile} NAME_WE)
  SET(target ${file})
  ADD_EXECUTABLE(${target} ${ccfile})
  SET_TARGET_PROPERTIES(${target} PROPERTIES OUTPUT_NAME ${file})
  DEAL_II_SETUP_TARGET(${target})
  IF(PVS-Analysis)
    pvs_studio_add_target(TARGET analyze_${target} ALL
                          ANALYZE ${target}
                          OUTPUT FORMAT errorfile
                          CXX_FLAGS ${DEAL_II_CXX_FLAGS}
                          LOG ${target}.plog
                          CONFIG "../PVS-Studio.cfg")
  ENDIF()
ENDFOREACH()
