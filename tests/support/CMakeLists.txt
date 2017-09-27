ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)

INCLUDE_DIRECTORIES(include)
SET(src_boost_test
#  libs/boost/test/src/cpp_main.cpp
  libs/boost/test/src/compiler_log_formatter.cpp
  libs/boost/test/src/debug.cpp
  libs/boost/test/src/decorator.cpp
  libs/boost/test/src/execution_monitor.cpp
  libs/boost/test/src/framework.cpp
  libs/boost/test/src/junit_log_formatter.cpp
  libs/boost/test/src/plain_report_formatter.cpp
  libs/boost/test/src/progress_monitor.cpp
  libs/boost/test/src/results_collector.cpp
  libs/boost/test/src/results_reporter.cpp
  libs/boost/test/src/test_main.cpp
  libs/boost/test/src/test_tools.cpp
  libs/boost/test/src/test_tree.cpp
  libs/boost/test/src/unit_test_log.cpp
  libs/boost/test/src/unit_test_main.cpp
  libs/boost/test/src/unit_test_monitor.cpp
  libs/boost/test/src/unit_test_parameters.cpp
  libs/boost/test/src/xml_log_formatter.cpp
  libs/boost/test/src/xml_report_formatter.cpp
  )

ADD_LIBRARY(boost_test SHARED ${src_boost_test})
