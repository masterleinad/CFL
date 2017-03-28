CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT (boost_test)

# check for a sufficiently recent compiler
INCLUDE(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
CHECK_CXX_COMPILER_FLAG("-std=c++1z" COMPILER_SUPPORTS_CXX1Z)
IF(COMPILER_SUPPORTS_CXX17)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
ELSEIF(COMPILER_SUPPORTS_CXX1Z)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z")
ELSE()
  MESSAGE(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
ENDIF()

ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)

INCLUDE_DIRECTORIES(../../../include)
SET(src_boost_test
#  src/cpp_main.cpp
  src/compiler_log_formatter.cpp
  src/debug.cpp                   
  src/decorator.cpp              
  src/execution_monitor.cpp    
  src/framework.cpp             
  src/junit_log_formatter.cpp  
  src/plain_report_formatter.cpp
  src/progress_monitor.cpp
  src/results_collector.cpp
  src/results_reporter.cpp
  src/test_main.cpp
  src/test_tools.cpp
  src/test_tree.cpp
  src/unit_test_log.cpp
  src/unit_test_main.cpp
  src/unit_test_monitor.cpp
  src/unit_test_parameters.cpp
  src/xml_log_formatter.cpp
  src/xml_report_formatter.cpp
  )

ADD_LIBRARY(boost_test SHARED ${src_boost_test})
