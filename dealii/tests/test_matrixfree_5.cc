///////
#define BOOST_TEST_MODULE TMOD_MATRIXFREE_5_H
#define BOOST_TEST_DYN_LINK
#include "test_matrixfree.h"
//////////

//// Test case ProdFEObjSameType
// Type: Positive test case
// Coverage: following classes - ProductFEFunctions
// Checks for:
// 1. Behavior for overloaded operator "*" when types of operands is exactly same
//    This is done by "invoking" operator overloading for several combinations which can
// conclude that:
//    - all possible ways to multiply FE functions is tested
//    - result of ProductFEFunction is of correct type
//    There are no check on values since they are available from dealii library function
// results which we ignore in out test
// TBD: It only tests for generation of form. not their types and result e.g. result of
// multiplication
template <template <int, int, unsigned int> typename FEFuncType, int rank, int dim,
          unsigned int idx>
void
prodfe_same_types()
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

  // multiplication of fefunction and prodfefunction amongst themselves
  auto prod1 = fe_function1 * fe_function2;
  auto prod2 = fe_function2 * fe_function1;

  // Sum and diff of products, its return type is to be discussed
  auto sum1 = prod1 + prod2;
  auto minus1 = prod1 - prod2;

  auto sum4 = fe_function1 + fe_function4 + fe_function5;

  auto prod3 = fe_function2 * fe_function1 * fe_function3;

  auto prod4 = fe_function1 * fe_function4 * fe_function5;
  auto prod5 = fe_function4 * prod1 * fe_function5;
  auto prod6 = fe_function4 * fe_function5 * prod1;

  auto prod7 = prod1 * prod2;
  auto prod8 = prod1 * prod2 * fe_function1;
  auto prod9 = fe_function2 * prod1 * prod2;
  auto prod10 = prod6 * prod8;

  auto prod11 = fe_function3 * fe_function4;
  auto prod12 = prod11 * prod10;

  // multiplciation of fefunction and prodfefunction with scalar
  auto prod13 = fe_function1 * 2;
  auto prod14 = 2 * fe_function1;
  auto prod15 = fe_function2 * 2.0;
  auto prod16 = 2.0 * fe_function2;
  auto prod17 = prod1 * prod15 * 2;
  auto prod18 = 2 * prod1 * prod15;

  auto prod19 = prod7 * 2.0;
  auto prod20 = 2.0 * prod7;

  auto prod21 = 2 * prod2 * prod5 * prod8 * prod10 * prod15; // something complicated

  auto prod22 = fe_function3 * true;  // this is stupid but accepted
  auto prod23 = false * fe_function3; // this is stupid but accepted
  auto prod24 = 'c' * fe_function3;   // this is stupid but accepted

  auto prodx = prod21 - prod22;
}

template <int i>
struct ProdFEfunctor
{
  static constexpr FEFunction_s obj_comb[9] = {
    // {rank, dim, index, scalar-factor}
    { 0, 1, 1, 0.1 }, { 0, 2, 1, 0.2 }, { 0, 3, 1, 0.3 }, { 1, 1, 1, 0.4 }, { 1, 2, 1, 0.5 },
    { 1, 3, 1, 0.6 }, { 2, 1, 1, 0.7 }, { 2, 2, 1, 0.8 }, { 2, 3, 1, 0.9 }
  };

  static void
  run()
  {
    prodfe_same_types<FEFunction, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    prodfe_same_types<FEDivergence, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    prodfe_same_types<FESymmetricGradient, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    prodfe_same_types<FECurl,obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index>();
    prodfe_same_types<FEGradient, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    prodfe_same_types<FELaplacian, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    prodfe_same_types<FEDiagonalHessian, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    prodfe_same_types<FEHessian, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    prodfe_same_types<FEFunctionInteriorFace, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    prodfe_same_types<FEFunctionExteriorFace, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    prodfe_same_types<FENormalGradientInteriorFace, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    prodfe_same_types<FENormalGradientExteriorFace, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
  }
};
BOOST_AUTO_TEST_CASE(ProdFEObjSameType)
{
  for_<0, 9>::run<ProdFEfunctor>();
}

//// Test case ProdFEObjDiffType
// Type: Positive test case
// Coverage: following classes - ProductFEFunctions
// Checks for:
// 1. Behavior for overloaded operator "*" when types of operands is different but tensor rank is
// equal
//    This is done by "invoking" operator overloading for several combinations which can
// conclude that:
//    - all possible ways to multiply FE functions is tested
//    - result of ProductFEFunction is of correct type
//    There are no check on values since they are available from dealii library function
// results which we ignore in out test
// Remarks:
// See remark 1 from SumFEObjDiffType
template <typename T>
void
prodfe_diff_types(const T& type1, const T& type2)
{
  auto prod1 = type1 * type2;
  auto prod2 = type2 * type1;
  auto prod3 = prod1 * prod2;
  auto prod4 = prod1 * prod3;
}

BOOST_AUTO_TEST_CASE(ProdFEObjDiffType)
{
  const int rank = 2;
  const int dim = 2;
  const int fe_no = 10; // random

  FEFunction<rank, dim, fe_no> F("test_obj");
  auto FG = grad(F);
  auto FGD = div(grad(F));
  auto FD = div(F);

  prodfe_diff_types(F, F);
  prodfe_diff_types(FG, FG);
  prodfe_diff_types(FD, FD);
}
