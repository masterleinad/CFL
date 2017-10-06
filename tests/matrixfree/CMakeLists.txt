INCLUDE_DIRECTORIES(../support/include) #for boost test
SET(TEST_TARGET ${TARGET})
SET(TEST_LIBRARIES ${CMAKE_BINARY_DIR}/tests/support/libboost_test.so)
DEAL_II_PICKUP_TESTS()
