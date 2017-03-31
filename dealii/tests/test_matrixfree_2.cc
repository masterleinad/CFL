///////
#define BOOST_TEST_MODULE TMOD_MATRIXFREE_2_H
#define BOOST_TEST_DYN_LINK
#include "test_matrixfree.h"
//////////

//// Test case FEObjCreation
// Type: Positive test case
// Coverage: following classes - FEFunction, FEDivergence, FELiftDivergence,
// FESymmetricGradient,FECurl
//                       FEGradient, FELaplacian, FEDiagonalHessian,
//             FEHessain
//       following functions - div(), grad()
// Checks for:
// 1. Template instantiation with several combinations of (rank,dim,index) as per  variable obj_comb
// 2. Object creation in each case, and basic state check for the object
template <int i>
struct FEfunctor
{
  static constexpr FEFunction_s obj_comb[9] = {
    // {rank, dim, index, scalar-factor}
    { 0, 1, 1, 0.1 }, { 0, 2, 1, 0.2 }, { 0, 3, 1, 0.3 }, { 1, 1, 1, 0.4 }, { 1, 2, 1, 0.5 },
    { 1, 3, 1, 0.6 }, { 2, 1, 1, 0.7 }, { 2, 2, 1, 0.8 }, { 2, 3, 1, 0.9 }
  };

  static void
  run()
  {
    // TBD: Change to avoid duplication??

    // Check possible constructions and basic state of objects
    // FEFunction<obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index> test_fe_fun_obj1_1; //default
    // const
    FEFunction<obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index> test_fe_fun_obj1_2(
      "test_fe_fun_obj1_2"); // one form of const
    FEFunction<obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index> test_fe_fun_obj1(
      "test_fe_fun_obj1", obj_comb[i].scalar_factor); // second form of const
    // Check basic state of object
    BOOST_TEST(test_fe_fun_obj1.index == obj_comb[i].index);
    BOOST_TEST(test_fe_fun_obj1.name() == "test_fe_fun_obj1");
    BOOST_TEST(test_fe_fun_obj1.scalar_factor == obj_comb[i].scalar_factor);

    auto test_grad_obj = grad(test_fe_fun_obj1);
    BOOST_TEST(test_grad_obj.index == test_fe_fun_obj1.index);
    BOOST_TEST(test_grad_obj.name() == test_fe_fun_obj1.name());
    BOOST_TEST(test_grad_obj.scalar_factor == test_fe_fun_obj1.scalar_factor);

#if 0 // TBD The construction fails due to no matching default constructor of base class. Discuss
    // why the constructor
    //of this class is different than others
    auto test_hessian_obj = grad(test_grad_obj);
    BOOST_TEST(test_hessian_obj.index == test_grad_obj.index);
    BOOST_TEST(test_hessian_obj.name() == test_grad_obj.name());
    BOOST_TEST(test_hessian_obj.scalar_factor == test_grad_obj.scalar_factor);
#endif

    auto test_div_obj = div(test_fe_fun_obj1);
    BOOST_TEST(test_div_obj.index == test_fe_fun_obj1.index);
    BOOST_TEST(test_div_obj.name() == test_fe_fun_obj1.name());
    BOOST_TEST(test_div_obj.scalar_factor == test_fe_fun_obj1.scalar_factor);

    auto test_laplacian_obj = div(test_grad_obj);
    BOOST_TEST(test_laplacian_obj.index == test_grad_obj.index);
    BOOST_TEST(test_laplacian_obj.name() == test_grad_obj.name());
    BOOST_TEST(test_laplacian_obj.scalar_factor == test_grad_obj.scalar_factor);
  }
};

BOOST_AUTO_TEST_CASE(FEObjCreation)
{
  // Check some combinations of Test objects
  for_<0, 9>::run<FEfunctor>();
}
