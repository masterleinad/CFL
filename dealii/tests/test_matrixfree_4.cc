///////
#define BOOST_TEST_MODULE TMOD_MATRIXFREE_4_H
#define BOOST_TEST_DYN_LINK
#include "test_matrixfree.h"
//////////

//// Test case SumFEObjSameType
// Type: Positive test case
// Coverage: following classes - SumFEFunctions
// Checks for:
// 1. Behavior for overloaded operator "+" when types of operands is exactly same
//    This is done by "invoking" operator overloading for several combinations which can conclude
//    that:
//    - all possible ways to sum FE functions is tested
//    - result of SumFEFunction is of correct type
//    There are no check on values since they are available from dealii library function results
//    which we ignore in out test
// Does not cover:
//     FELiftDivergence since it is different from others and does not support (+) operation yet
template <template <int, int, unsigned int> typename FEFuncType, int rank, int dim,
          unsigned int idx>
void
sumfe_same_types()
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

  auto sum1 = fe_function1 + fe_function2;
  auto sum2 = fe_function2 + fe_function1;

  auto sum3 = fe_function2 + fe_function1 + fe_function3;
  auto sum4 = fe_function1 + fe_function4 + fe_function5;
  auto sum5 = fe_function4 + sum1 + fe_function5;
  auto sum6 = fe_function4 + fe_function5 + sum1;
  auto sum7 = sum1 + sum2;
  auto sum8 = sum1 + sum2 + fe_function1;
  auto sum9 = fe_function2 + sum1 + sum2;
  auto sum10 = sum6 + sum8;
  auto sum11 = fe_function3 + fe_function4;
  auto sum12 = sum11 + sum10;

  // Prod of sum
  auto prod1 = sum1 * sum2;

  auto sum13 = sum2 + sum5 + sum8 + sum10 + sum12; // something complicated
}

template <int i>
struct SumFEfunctor
{
  static constexpr FEFunction_s obj_comb[9] = {
    // {rank, dim, index, scalar-factor}
    { 0, 1, 1, 0.1 }, { 0, 2, 1, 0.2 }, { 0, 3, 1, 0.3 }, { 1, 1, 1, 0.4 }, { 1, 2, 1, 0.5 },
    { 1, 3, 1, 0.6 }, { 2, 1, 1, 0.7 }, { 2, 2, 1, 0.8 }, { 2, 3, 1, 0.9 }
  };

  static void
  run()
  {
    sumfe_same_types<FEFunction, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    sumfe_same_types<FEDivergence, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    sumfe_same_types<FESymmetricGradient, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    sumfe_same_types<FECurl, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    sumfe_same_types<FEGradient, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    sumfe_same_types<FELaplacian, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    sumfe_same_types<FEDiagonalHessian, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    sumfe_same_types<FEHessian, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    sumfe_same_types<FEFunctionInteriorFace,
                     obj_comb[i].rank,
                     obj_comb[i].dim,
                     obj_comb[i].index>();
    sumfe_same_types<FEFunctionExteriorFace,
                     obj_comb[i].rank,
                     obj_comb[i].dim,
                     obj_comb[i].index>();
    sumfe_same_types<FENormalGradientInteriorFace,
                     obj_comb[i].rank,
                     obj_comb[i].dim,
                     obj_comb[i].index>();
    sumfe_same_types<FENormalGradientExteriorFace,
                     obj_comb[i].rank,
                     obj_comb[i].dim,
                     obj_comb[i].index>();
  }
};
BOOST_FIXTURE_TEST_CASE(SumFEObjSameType, FEFixture)
{
  for_<0, 9>::run<SumFEfunctor>();

  // auto sumf = fe_function_scalar1 + fe_function_scalar2; //TBD why does this not hit
  // static_assertion although compilation fails?
}

//// Test case SumFEObjDiffType
// Type: Positive test case
// Coverage: following classes - SumFEFunctions
// Checks for:
// 1. Behavior for overloaded operator "+" when types of operands is different but tensor rank is
// equal
//    This is done by "invoking" operator overloading for several combinations which can conclude
//    that:
//    - all possible ways to sum FE functions is tested
//    - result of SumFEFunction is of correct type
//    There are no check on values since they are available from dealii library function results
//    which we ignore in out test
// Remarks:
// 1. How are the combinations of different types chosen?
//    See that grad() returns FEFunction of rank+1, while div returns FEFunction of rank-1
//    The state machine would be (< and > should be seen as arrow termination):
/*
                 FE
                | |
               /   \
            n |     | n
             /       \
            <         >
      (+1)Grad -------> Div(-1)
       |   >
       |   |
       |___|
*/
//    i.e. valid state transitions are
//      operation     output tensor rank   supported
//        F           n                      Y
//        GF          n+1                    Y
//        DGF         n                      Y
//        GGF         n+2                    N
//        DGGF        n+1                    N
//        DF          n-1                    Y
// where e.g. DGF => div(grad(FEFunction F))
template <typename T>
void
sumfe_diff_types(const T& type1, const T& type2)
{
  auto sum1 = type1 + type2;
  auto sum2 = type2 + type1;
  auto sum3 = sum1 + sum2;
  auto sum4 = sum1 + sum3;
}

BOOST_AUTO_TEST_CASE(SumFEObjDiffType)
{
  const int rank = 2;
  const int dim = 2;
  const int fe_no = 10; // random

  FEFunction<rank, dim, fe_no> F("test_obj");
  auto GF = grad(F);
  auto DGF = div(grad(F));
  auto DF = div(F);

  sumfe_diff_types(F, F);
  sumfe_diff_types(GF, GF);
  sumfe_diff_types(DF, DF);
}
