///////
#define BOOST_TEST_MODULE TMOD_MATRIXFREE_1_H
#define BOOST_TEST_DYN_LINK
#include "test_matrixfree.h"
//////////

//// Test case TestObjCreation
// Type: Positive test case
// Coverage: following classes - TestFunction, TestDivergence, TestSymmetricGradient, TestCurl,
// TestGradient, TestHessian, TestFunctionInteriorFace, TestFunctionExteriorFace, TestNormalGradientInteriorFace,
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
  typedef map<string, pair<bool, bool>>
    comb_map_t; // Testxxx class name, (integrate_value, integrate_gradient)
  static comb_map_t i_comb;

  static void
  run()
  {
    // TBD: Change to avoid duplication??
    TestFunction<obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index> test_obj1;
    // Check basic state of object
    BOOST_TEST(test_obj1.index == obj_comb[i].index);
    BOOST_TEST(test_obj1.scalar_valued == obj_comb[i].scalar_valued);
    BOOST_TEST(test_obj1.integrate_value == i_comb["TestFunction"].first);
    BOOST_TEST(test_obj1.integrate_gradient == i_comb["TestFunction"].second);

    TestDivergence<obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index> test_obj2;
    // Check basic state of object
    BOOST_TEST(test_obj2.index == obj_comb[i].index);
    BOOST_TEST(test_obj2.scalar_valued == obj_comb[i].scalar_valued);
    BOOST_TEST(test_obj2.integrate_value == i_comb["TestDivergence"].first);
    BOOST_TEST(test_obj2.integrate_gradient == i_comb["TestDivergence"].second);

    TestSymmetricGradient<obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index> test_obj3;
    // Check basic state of object
    BOOST_TEST(test_obj3.index == obj_comb[i].index);
    BOOST_TEST(test_obj3.scalar_valued == obj_comb[i].scalar_valued);
    BOOST_TEST(test_obj3.integrate_value == i_comb["TestSymmetricGradient"].first);
    BOOST_TEST(test_obj3.integrate_gradient == i_comb["TestSymmetricGradient"].second);

    TestCurl<obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index> test_obj4;
    // Check basic state of object
    BOOST_TEST(test_obj4.index == obj_comb[i].index);
    BOOST_TEST(test_obj4.scalar_valued == obj_comb[i].scalar_valued);
    BOOST_TEST(test_obj4.integrate_value == i_comb["TestCurl"].first);
    BOOST_TEST(test_obj4.integrate_gradient == i_comb["TestCurl"].second);

    TestGradient<obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index> test_obj5;
    // Check basic state of object
    BOOST_TEST(test_obj5.index == obj_comb[i].index);
    BOOST_TEST(test_obj5.scalar_valued == obj_comb[i].scalar_valued);
    BOOST_TEST(test_obj5.integrate_value == i_comb["TestGradient"].first);
    BOOST_TEST(test_obj5.integrate_gradient == i_comb["TestGradient"].second);

    TestHessian<obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index> test_obj6;
    // Check basic state of object
    BOOST_TEST(test_obj6.index == obj_comb[i].index);
    BOOST_TEST(test_obj6.scalar_valued == obj_comb[i].scalar_valued);
    BOOST_TEST(test_obj6.integrate_value == i_comb["TestHessian"].first);
    BOOST_TEST(test_obj6.integrate_gradient == i_comb["TestHessian"].second);

    TestFunctionInteriorFace<obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index> test_obj7;
    // Check basic state of object
    BOOST_TEST(test_obj7.index == obj_comb[i].index);
    BOOST_TEST(test_obj7.scalar_valued == obj_comb[i].scalar_valued);
    BOOST_TEST(test_obj7.integrate_value == i_comb["TestFunctionInteriorFace"].first);
    BOOST_TEST(test_obj7.integrate_gradient == i_comb["TestFunctionInteriorFace"].second);

    TestFunctionExteriorFace<obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index> test_obj8;
    // Check basic state of object
    BOOST_TEST(test_obj8.index == obj_comb[i].index);
    BOOST_TEST(test_obj8.scalar_valued == obj_comb[i].scalar_valued);
    BOOST_TEST(test_obj8.integrate_value == i_comb["TestFunctionExteriorFace"].first);
    BOOST_TEST(test_obj8.integrate_gradient == i_comb["TestFunctionExteriorFace"].second);

    TestNormalGradientInteriorFace<obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index> test_obj9;
    // Check basic state of object
    BOOST_TEST(test_obj9.index == obj_comb[i].index);
    BOOST_TEST(test_obj9.scalar_valued == obj_comb[i].scalar_valued);
    BOOST_TEST(test_obj9.integrate_value == i_comb["TestNormalGradientInteriorFace"].first);
    BOOST_TEST(test_obj9.integrate_gradient == i_comb["TestNormalGradientInteriorFace"].second);

    TestNormalGradientExteriorFace<obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index> test_obj10;
    // Check basic state of object
    BOOST_TEST(test_obj10.index == obj_comb[i].index);
    BOOST_TEST(test_obj10.scalar_valued == obj_comb[i].scalar_valued);
    BOOST_TEST(test_obj10.integrate_value == i_comb["TestNormalGradientExteriorFace"].first);
    BOOST_TEST(test_obj10.integrate_gradient == i_comb["TestNormalGradientExteriorFace"].second);

    //We dont need to test anything else here
    auto test_obj_grad = grad(test_obj1);
    auto test_hessian_obj = grad(test_obj_grad);
    auto test_div_obj = div(test_obj1);

  }
};

// ugly initialization of static member
template <int i>
typename TestDatafunctor<i>::comb_map_t TestDatafunctor<i>::i_comb = {
  { "TestFunction", { true, false } },          { "TestDivergence", { false, true } },
  { "TestSymmetricGradient", { false, true } }, { "TestCurl", { false, true } },
  { "TestGradient", { false, true } },          { "TestHessian", { false, false } },
  { "TestFunctionInteriorFace", { true, false }},{ "TestFunctionExteriorFace", { true, false } },
  { "TestNormalGradientInteriorFace", { true, false } }, { "TestNormalGradientExteriorFace", { true, false } }
};

BOOST_AUTO_TEST_CASE(TestObjCreation)
{
  // Check some combinations of Test objects
  for_<0, 9>::run<TestDatafunctor>();
}

////////////////////////////////////////////////////////////////////////////
