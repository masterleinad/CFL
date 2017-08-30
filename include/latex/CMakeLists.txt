CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)
PROJECT(cfllatex)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/..)

FILE(GLOB sources *.cc)
GET_FILENAME_COMPONENT(prefix ${CMAKE_CURRENT_SOURCE_DIR} NAME)

INCLUDE(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++1y" COMPILER_SUPPORTS_CXX1Y)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
IF(COMPILER_SUPPORTS_CXX14)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
ELSEIF(COMPILER_SUPPORTS_CXX1Y)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")
ELSE()
  MESSAGE(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
ENDIF()

OPTION(PVS-Analysis "Use static code analyzer PVS-Studio?" OFF)

IF(PVS-Analysis)
  INCLUDE(../PVS-Studio.cmake)
ENDIF()
FOREACH(ccfile ${sources})
  GET_FILENAME_COMPONENT(file ${ccfile} NAME_WE)
  SET(target ${file})
  ADD_EXECUTABLE(${target} ${ccfile})
  SET_TARGET_PROPERTIES(${target} PROPERTIES OUTPUT_NAME ${file})
  IF(PVS-Analysis)
    pvs_studio_add_target(TARGET analyze_${target} ALL
                          ANALYZE ${target}
                          OUTPUT FORMAT errorfile
                          CXX_FLAGS ${CMAKE_CXX_FLAGS}
                          LOG ${target}.plog
                          CONFIG "../PVS-Studio.cfg")
  ENDIF()
ENDFOREACH()
