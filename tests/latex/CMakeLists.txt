SET(TEST_TARGET ${TARGET})
SET(TEST_LIBRARIES  ${CMAKE_SOURCE_DIR}/third_party/libs/boost/test/libboost_test.so)
DEAL_II_PICKUP_TESTS()
