CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT (boost_test)

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
