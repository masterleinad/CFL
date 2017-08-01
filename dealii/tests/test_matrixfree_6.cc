///////
#define BOOST_TEST_MODULE TMOD_MATRIXFREE_6_H
#define BOOST_TEST_DYN_LINK
#include "test_matrixfree.h"
//////////

//// Test case MinusFEObjSameType
// Type: Positive test case
// Coverage: following classes - SumFEFunctions
// Checks for:
// 1. Behavior for overloaded operator "-" when types of operands is exactly same
//    This is done by "invoking" operator overloading for several combinations which can conclude
//    that:
//    - all possible ways to sum FE functions is tested
//    - result of SumFEFunction is of correct type
//    There are no check on values since they are available from dealii library function results
//    which we ignore in out test
// Does not cover:
//     FELiftDivergence since it is different from others and does not support (-) operation yet
template <template <int, int, unsigned int> typename FEFuncType, int rank, int dim,
          unsigned int idx>
void
minusfe_same_types()
{
  constexpr unsigned int idx1 = idx;
  constexpr unsigned int idx2 = idx + 2;
  constexpr unsigned int idx3 = idx + 4;
  constexpr unsigned int idx4 = idx + 6;
  constexpr unsigned int idx5 = idx + 8;
  constexpr int max_idx = idx5;

  bitset<max_idx> bs;

  FEFuncType<rank, dim, idx1> fe_function1("test_fe1");
  FEFuncType<rank, dim, idx2> fe_function2("test_fe2");
  FEFuncType<rank, dim, idx3> fe_function3("test_fe3");
  FEFuncType<rank, dim, idx4> fe_function4("test_fe4");
  FEFuncType<rank, dim, idx5> fe_function5("test_fe5");

  auto minus1 = fe_function1 - fe_function2;
  auto minus2 = fe_function2 - fe_function1;

  auto minus3 = fe_function2 - fe_function1 - fe_function3;
  auto minus4 = fe_function1 - fe_function4 - fe_function5;
  auto minus5 = fe_function4 - minus1 - fe_function5;
  auto minus6 = fe_function4 - fe_function5 - minus1;
  auto minus7 = minus1 - minus2;
  auto minus8 = minus1 - minus2 - fe_function1;
  auto minus9 = fe_function2 - minus1 - minus2;
  auto minus10 = minus6 - minus8;
  auto minus11 = fe_function3 - fe_function4;
  auto minus12 = minus11 - minus10;

  // Prod of minus
  auto prod1 = minus1 * minus2;

  auto minus13 = minus2 - minus5 - minus8 - minus10 - minus12; // something complicated
}

template <int i>
struct MinusFEfunctor
{
  static constexpr FEFunction_s obj_comb[9] = {
    // {rank, dim, index, scalar-factor}
    { 0, 1, 1, 0.1 }, { 0, 2, 1, 0.2 }, { 0, 3, 1, 0.3 }, { 1, 1, 1, 0.4 }, { 1, 2, 1, 0.5 },
    { 1, 3, 1, 0.6 }, { 2, 1, 1, 0.7 }, { 2, 2, 1, 0.8 }, { 2, 3, 1, 0.9 }
  };

  static void
  run()
  {
    minusfe_same_types<FEFunction, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    minusfe_same_types<FEDivergence, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    minusfe_same_types<FESymmetricGradient, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    minusfe_same_types<FECurl, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    minusfe_same_types<FEGradient, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    minusfe_same_types<FELaplacian, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    minusfe_same_types<FEDiagonalHessian, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    minusfe_same_types<FEHessian, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    minusfe_same_types<FEFunctionInteriorFace,
                       obj_comb[i].rank,
                       obj_comb[i].dim,
                       obj_comb[i].index>();
    minusfe_same_types<FEFunctionExteriorFace,
                       obj_comb[i].rank,
                       obj_comb[i].dim,
                       obj_comb[i].index>();
    minusfe_same_types<FENormalGradientInteriorFace,
                       obj_comb[i].rank,
                       obj_comb[i].dim,
                       obj_comb[i].index>();
    minusfe_same_types<FENormalGradientExteriorFace,
                       obj_comb[i].rank,
                       obj_comb[i].dim,
                       obj_comb[i].index>();
  }
};
BOOST_FIXTURE_TEST_CASE(MinusFEObjSameType, FEFixture)
{
  for_<0, 9>::run<MinusFEfunctor>();
}

//// Test case MinusFEObjDiffType
// Type: Positive test case
// Coverage: following classes - SumFEFunctions
// Checks for:
// 1. Behavior for overloaded operator "-" when types of operands is different but tensor rank is
// Remarks:
// 1. Conceptually similar to SumFEObjDiffType, see that for more details
template <typename T>
void
minusfe_diff_types(const T& type1, const T& type2)
{
  auto minus1 = type1 - type2;
  auto minus2 = type2 - type1;
  auto minus3 = minus1 - minus2;
  auto minus4 = minus1 - minus3;
}

BOOST_AUTO_TEST_CASE(MinusFEObjDiffType)
{
  const int rank = 2;
  const int dim = 2;
  const int fe_no = 10; // random

  FEFunction<rank, dim, fe_no> F("test_obj");
  auto GF = grad(F);
  auto DGF = div(grad(F));
  auto DF = div(F);

  minusfe_diff_types(F, F);
  minusfe_diff_types(GF, GF);
  minusfe_diff_types(DF, DF);
}
