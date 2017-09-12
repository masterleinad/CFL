CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)
FIND_PACKAGE(deal.II 9.0.0 REQUIRED
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} $ENV{DEAL_II_DIR}
  )

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(CFL)

INCLUDE(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++1z" COMPILER_SUPPORTS_CXX1Z)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
IF(COMPILER_SUPPORTS_CXX1Z)
  SET(DEAL_II_CXX_FLAGS "${DEAL_II_CXX_FLAGS} -std=c++1z")
ELSEIF(COMPILER_SUPPORTS_CXX1/)
  SET(DEAL_II_CXX_FLAGS "${DEAL_II_CXX_FLAGS} -std=c++17")
ELSE()
  MESSAGE(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
ENDIF()

OPTION(PVS-Analysis "Use static code analyzer PVS-Studio for applications?" OFF)

INCLUDE_DIRECTORIES(include)
FILE(GLOB SOURCES_CFL "sources/cfl/*.cc")
FILE(GLOB SOURCES_LATEX "sources/latex/*.cc")
FILE(GLOB SOURCES_MATRIXFREE "sources/matrixfree/*.cc")
FILE(GLOB SOURCES_MESHWORKER "sources/meshworker/*.cc")
ADD_LIBRARY(cfl
    ${SOURCES_CFL}
    ${SOURCES_LATEX}
    ${SOURCES_MATRIXFREE}
    ${SOURCES_MESHWORKER}
    )
DEAL_II_SETUP_TARGET(cfl)
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

ADD_SUBDIRECTORY(applications)

ADD_SUBDIRECTORY(doc)

# Enable testing and descent into tests/ subdirectory:
INCLUDE_DIRECTORIES(third_party/include) #for boost test
ENABLE_TESTING()
ADD_SUBDIRECTORY(tests)
