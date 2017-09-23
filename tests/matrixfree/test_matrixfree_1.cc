///////
#define BOOST_TEST_MODULE TMOD_MATRIXFREE_1_H
#define BOOST_TEST_DYN_LINK
#include "test_matrixfree.h"
//////////

//// Test case TestObjCreation
// Type: Positive test case
// Coverage: following classes - TestFunction, TestDivergence, TestSymmetricGradient, TestCurl,
// TestGradient, TestHessian, TestFunctionInteriorFace, TestFunctionExteriorFace,
// TestNormalGradientInteriorFace,
// TestNormalGradientExteriorFace
// Checks for:
// 1. Template instantiation with several combinations of (rank,dim,index) as per  variable obj_comb
// 2. Object creation in each case, and basic state check for the object
template <int i>
struct TestDatafunctor
{
  static constexpr TestData_s obj_comb[9] = {
    // {rank, dim, index, scalar_valued}
    { 0, 1, 1, 1 }, { 0, 2, 1, 1 }, { 0, 3, 1, 1 }, { 1, 1, 1, 0 }, { 1, 2, 1, 0 },
    { 1, 3, 1, 0 }, { 2, 1, 1, 0 }, { 2, 2, 1, 0 }, { 2, 3, 1, 0 }
  };
  typedef struct
  {
    bool first;
    bool second;
    bool third;
    bool fourth;
  } intg_flags;

  // Testxxx class name, (integration_flags.value,value_exterior,
  //     integration_flags.gradient, gradient_exterior)
  typedef map<string, intg_flags> comb_map_t;
  static comb_map_t i_comb;

  static void
  run()
  {
    // TBD: Change to avoid duplication??
    Base::TestFunction<obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index> test_obj1;
    // Check basic state of object
    BOOST_TEST(test_obj1.index == obj_comb[i].index);
    BOOST_TEST(test_obj1.scalar_valued == obj_comb[i].scalar_valued);
    BOOST_TEST(test_obj1.integration_flags.value == i_comb["TestFunction"].first);
    BOOST_TEST(test_obj1.integration_flags.value_exterior == i_comb["TestFunction"].second);
    BOOST_TEST(test_obj1.integration_flags.gradient == i_comb["TestFunction"].third);
    BOOST_TEST(test_obj1.integration_flags.gradient_exterior == i_comb["TestFunction"].fourth);

    TestDivergence<obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index> test_obj2;
    // Check basic state of object
    BOOST_TEST(test_obj2.index == obj_comb[i].index);
    BOOST_TEST(test_obj2.scalar_valued == obj_comb[i].scalar_valued);
    BOOST_TEST(test_obj2.integration_flags.value == i_comb["TestDivergence"].first);
    BOOST_TEST(test_obj2.integration_flags.value_exterior == i_comb["TestDivergence"].second);
    BOOST_TEST(test_obj2.integration_flags.gradient == i_comb["TestDivergence"].third);
    BOOST_TEST(test_obj2.integration_flags.gradient_exterior == i_comb["TestDivergence"].fourth);

    TestSymmetricGradient<obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index> test_obj3;
    // Check basic state of object
    BOOST_TEST(test_obj3.index == obj_comb[i].index);
    BOOST_TEST(test_obj3.scalar_valued == obj_comb[i].scalar_valued);
    BOOST_TEST(test_obj3.integration_flags.value == i_comb["TestSymmetricGradient"].first);
    BOOST_TEST(test_obj3.integration_flags.value_exterior ==
               i_comb["TestSymmetricGradient"].second);
    BOOST_TEST(test_obj3.integration_flags.gradient == i_comb["TestSymmetricGradient"].third);
    BOOST_TEST(test_obj3.integration_flags.gradient_exterior ==
               i_comb["TestSymmetricGradient"].fourth);

    TestCurl<obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index> test_obj4;
    // Check basic state of object
    BOOST_TEST(test_obj4.index == obj_comb[i].index);
    BOOST_TEST(test_obj4.scalar_valued == obj_comb[i].scalar_valued);
    BOOST_TEST(test_obj4.integration_flags.value == i_comb["TestCurl"].first);
    BOOST_TEST(test_obj4.integration_flags.value_exterior == i_comb["TestCurl"].second);
    BOOST_TEST(test_obj4.integration_flags.gradient == i_comb["TestCurl"].third);
    BOOST_TEST(test_obj4.integration_flags.gradient_exterior == i_comb["TestCurl"].fourth);

    TestGradient<obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index> test_obj5;
    // Check basic state of object
    BOOST_TEST(test_obj5.index == obj_comb[i].index);
    BOOST_TEST(test_obj5.scalar_valued == obj_comb[i].scalar_valued);
    BOOST_TEST(test_obj5.integration_flags.value == i_comb["TestGradient"].first);
    BOOST_TEST(test_obj5.integration_flags.value_exterior == i_comb["TestGradient"].second);
    BOOST_TEST(test_obj5.integration_flags.gradient == i_comb["TestGradient"].third);
    BOOST_TEST(test_obj5.integration_flags.gradient_exterior == i_comb["TestGradient"].fourth);

    TestHessian<obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index> test_obj6;
    // Check basic state of object
    BOOST_TEST(test_obj6.index == obj_comb[i].index);
    BOOST_TEST(test_obj6.scalar_valued == obj_comb[i].scalar_valued);
    BOOST_TEST(test_obj6.integration_flags.value == i_comb["TestHessian"].first);
    BOOST_TEST(test_obj6.integration_flags.value_exterior == i_comb["TestHessian"].second);
    BOOST_TEST(test_obj6.integration_flags.gradient == i_comb["TestHessian"].third);
    BOOST_TEST(test_obj6.integration_flags.gradient_exterior == i_comb["TestHessian"].fourth);

    TestFunctionInteriorFace<obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index> test_obj7;
    // Check basic state of object
    BOOST_TEST(test_obj7.index == obj_comb[i].index);
    BOOST_TEST(test_obj7.scalar_valued == obj_comb[i].scalar_valued);
    BOOST_TEST(test_obj7.integration_flags.value == i_comb["TestFunctionInteriorFace"].first);
    BOOST_TEST(test_obj7.integration_flags.value_exterior ==
               i_comb["TestFunctionInteriorFace"].second);
    BOOST_TEST(test_obj7.integration_flags.gradient == i_comb["TestFunctionInteriorFace"].third);
    BOOST_TEST(test_obj7.integration_flags.gradient_exterior ==
               i_comb["TestFunctionInteriorFace"].fourth);

    TestFunctionExteriorFace<obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index> test_obj8;
    // Check basic state of object
    BOOST_TEST(test_obj8.index == obj_comb[i].index);
    BOOST_TEST(test_obj8.scalar_valued == obj_comb[i].scalar_valued);
    BOOST_TEST(test_obj8.integration_flags.value == i_comb["TestFunctionExteriorFace"].first);
    BOOST_TEST(test_obj8.integration_flags.value_exterior ==
               i_comb["TestFunctionExteriorFace"].second);
    BOOST_TEST(test_obj8.integration_flags.gradient == i_comb["TestFunctionExteriorFace"].third);
    BOOST_TEST(test_obj8.integration_flags.gradient_exterior ==
               i_comb["TestFunctionExteriorFace"].fourth);

    TestNormalGradientInteriorFace<obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index> test_obj9;
    // Check basic state of object
    BOOST_TEST(test_obj9.index == obj_comb[i].index);
    BOOST_TEST(test_obj9.scalar_valued == obj_comb[i].scalar_valued);
    BOOST_TEST(test_obj9.integration_flags.value == i_comb["TestNormalGradientInteriorFace"].first);
    BOOST_TEST(test_obj9.integration_flags.value_exterior ==
               i_comb["TestNormalGradientInteriorFace"].second);
    BOOST_TEST(test_obj9.integration_flags.gradient ==
               i_comb["TestNormalGradientInteriorFace"].third);
    BOOST_TEST(test_obj9.integration_flags.gradient_exterior ==
               i_comb["TestNormalGradientInteriorFace"].fourth);

    TestNormalGradientExteriorFace<obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index> test_obj10;
    // Check basic state of object
    BOOST_TEST(test_obj10.index == obj_comb[i].index);
    BOOST_TEST(test_obj10.scalar_valued == obj_comb[i].scalar_valued);
    BOOST_TEST(test_obj10.integration_flags.value ==
               i_comb["TestNormalGradientExteriorFace"].first);
    BOOST_TEST(test_obj10.integration_flags.value_exterior ==
               i_comb["TestNormalGradientExteriorFace"].second);
    BOOST_TEST(test_obj10.integration_flags.gradient ==
               i_comb["TestNormalGradientExteriorFace"].third);
    BOOST_TEST(test_obj10.integration_flags.gradient_exterior ==
               i_comb["TestNormalGradientExteriorFace"].fourth);

    // We dont need to test anything else here
    auto test_obj_grad = grad(test_obj1);
    auto test_hessian_obj = grad(test_obj_grad);
    auto test_div_obj = div(test_obj1);
  }
};

// ugly initialization of static member
template <int i>
typename TestDatafunctor<i>::comb_map_t TestDatafunctor<i>::i_comb = {
  { "TestFunction", { true, false, false, false } },
  { "TestDivergence", { false, false, true, false } },
  { "TestSymmetricGradient", { false, false, true, false } },
  { "TestCurl", { false, false, true, false } },
  { "TestGradient", { false, false, true, false } },
  { "TestHessian", { false, false, false, false } },
  { "TestFunctionInteriorFace", { true, false, false, false } },
  { "TestFunctionExteriorFace", { false, true, false, false } },
  { "TestNormalGradientInteriorFace", { false, false, true, false } },
  { "TestNormalGradientExteriorFace", { false, false, false, true } }
};

BOOST_AUTO_TEST_CASE(TestObjCreation)
{
  // Check some combinations of Test objects
  for_<0, 9>::run<TestDatafunctor>();
}

////////////////////////////////////////////////////////////////////////////
