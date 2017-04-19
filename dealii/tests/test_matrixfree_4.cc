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
template <template <int, int, unsigned int> typename FEFuncType, int rank, int dim, unsigned int idx>
void
sumfe_same_types()
{
  constexpr unsigned int idx1 = idx;
  constexpr unsigned int idx2 = idx+2;
  constexpr unsigned int idx3 = idx+4;
  constexpr unsigned int idx4 = idx+6;
  constexpr unsigned int idx5 = idx+8;

  constexpr int max_idx = idx5;

  bitset<max_idx> bs;

  FEFuncType<rank, dim, idx1> fe_function1("test_fe1");
  FEFuncType<rank, dim, idx2> fe_function2("test_fe2");
  FEFuncType<rank, dim, idx3> fe_function3("test_fe3");
  FEFuncType<rank, dim, idx4> fe_function4("test_fe4");
  FEFuncType<rank, dim, idx5> fe_function5("test_fe5");



  auto sum1 = fe_function1 + fe_function2;
  check_FeOpComplete(sum1,bs,vector<unsigned int> {idx1,idx2});
  BOOST_TEST(bs.none());
  BOOST_TEST(sum1.n == 2);

  auto diff1 = fe_function1 - fe_function2;
  check_FeOpComplete(diff1,bs,vector<unsigned int> {idx1,idx2});
  BOOST_TEST(bs.none());
  BOOST_TEST(diff1.n == 2);

  auto sum2 = fe_function2 + fe_function1;
  check_FeOpComplete(sum2,bs,vector<unsigned int> {idx1,idx2});
  BOOST_TEST(bs.none());
  BOOST_TEST(sum2.n == 2);
  auto diff2 = fe_function2 - fe_function1;
  check_FeOpComplete(diff2,bs,vector<unsigned int> {idx1,idx2});
  BOOST_TEST(bs.none());
  BOOST_TEST(diff2.n == 2);

  auto sum3 = fe_function2 + fe_function1 + fe_function3;
  check_FeOpComplete(sum3,bs,vector<unsigned int> {idx1,idx2,idx3});
  BOOST_TEST(bs.none());
  BOOST_TEST(sum3.n == 3);
  auto diff3 = fe_function2 - fe_function1 - fe_function3;
  check_FeOpComplete(diff3,bs,vector<unsigned int> {idx1,idx2,idx3});
  BOOST_TEST(bs.none());
  BOOST_TEST(diff3.n == 3);

  auto sum4 = sum1 + fe_function4 + fe_function5;
  check_FeOpComplete(sum4,bs,vector<unsigned int> {idx1,idx2,idx4,idx5});
  BOOST_TEST(bs.none());
  BOOST_TEST(sum4.n == 4);
  auto diff4 = diff1 - fe_function4 - fe_function5;
  check_FeOpComplete(diff4,bs,vector<unsigned int> {idx1,idx2,idx4,idx5});
  BOOST_TEST(bs.none());
  BOOST_TEST(diff4.n == 4);

  //TBD: Discuss or note it down in doc..the order is not preserved
  //if fefunc + sum is performed
  auto sum5 = fe_function4 + sum1 + fe_function5;
  check_FeOpComplete(sum5,bs,vector<unsigned int> {idx1,idx2,idx4,idx5});
  BOOST_TEST(bs.none());
  BOOST_TEST(sum5.n == 4);
  auto diff5 = fe_function4 - diff1 - fe_function5;
  check_FeOpComplete(diff5,bs,vector<unsigned int> {idx1,idx2,idx4,idx5});
  BOOST_TEST(bs.none());
  BOOST_TEST(diff5.n == 4);

  auto sum6 = fe_function4 + fe_function5 + sum1;
  check_FeOpComplete(sum6,bs,vector<unsigned int> {idx1,idx2,idx4,idx5});
  BOOST_TEST(bs.none());
  BOOST_TEST(sum6.n == 4);
  auto diff6 = fe_function4 - fe_function5 - diff1;
  check_FeOpComplete(diff6,bs,vector<unsigned int> {idx1,idx2,idx4,idx5});
  BOOST_TEST(bs.none());
  BOOST_TEST(diff6.n == 4);

  auto sum7 = sum1 + sum2;
  check_FeOpComplete(sum7,bs,vector<unsigned int> {idx1,idx2});
  BOOST_TEST(bs.none());
  BOOST_TEST(sum7.n == 4);
  auto diff7 = diff1 - diff2;
  check_FeOpComplete(diff7,bs,vector<unsigned int> {idx1,idx2});
  BOOST_TEST(bs.none());
  BOOST_TEST(diff7.n == 4);

  auto sum8 = sum1 + sum2 + fe_function1;
  check_FeOpComplete(sum8,bs,vector<unsigned int> {idx1,idx2});
  BOOST_TEST(bs.none());
  BOOST_TEST(sum8.n == 5);
  auto diff8 = diff1 - diff2 - fe_function1;
  check_FeOpComplete(diff8,bs,vector<unsigned int> {idx1,idx2});
  BOOST_TEST(bs.none());
  BOOST_TEST(diff8.n == 5);

  auto sum9 = fe_function2 + sum1 + sum2;
  check_FeOpComplete(sum9,bs,vector<unsigned int> {idx1,idx2});
  BOOST_TEST(bs.none());
  BOOST_TEST(sum9.n == 5);
  auto diff9 = fe_function2 - diff1 - diff2;
  check_FeOpComplete(diff9,bs,vector<unsigned int> {idx1,idx2});
  BOOST_TEST(bs.none());
  BOOST_TEST(diff9.n == 5);

  auto sum10 = sum6 + sum8;
  check_FeOpComplete(sum10,bs,vector<unsigned int> {idx1,idx2,idx4,idx5});
  BOOST_TEST(bs.none());
  BOOST_TEST(sum10.n == 9);
  auto diff10 = diff6 - diff8;
  check_FeOpComplete(diff10,bs,vector<unsigned int> {idx1,idx2,idx4,idx5});
  BOOST_TEST(bs.none());
  BOOST_TEST(diff10.n == 9);

  //Check for unary minus combinations
  auto diff11 = -sum1;
  check_FeOpComplete(diff11,bs,vector<unsigned int> {idx1,idx2});
  BOOST_TEST(bs.none());
  BOOST_TEST(diff11.n == 2);

  auto diff12 = -sum2;
  check_FeOpComplete(diff12,bs,vector<unsigned int> {idx1,idx2});
  BOOST_TEST(bs.none());
  BOOST_TEST(diff12.n == 2);


  auto diff13 = -sum3;
  check_FeOpComplete(diff13,bs,vector<unsigned int> {idx1,idx2,idx3});
  BOOST_TEST(bs.none());
  BOOST_TEST(diff13.n == 3);

  auto diff14 = -sum4;
  check_FeOpComplete(diff14,bs,vector<unsigned int> {idx1,idx2,idx4,idx5});
  BOOST_TEST(bs.none());
  BOOST_TEST(diff14.n == 4);

  auto diff15 = -sum5;
  check_FeOpComplete(diff15,bs,vector<unsigned int> {idx1,idx2,idx4,idx5});
  BOOST_TEST(bs.none());
  BOOST_TEST(diff15.n == 4);

  auto diff16 = -sum6;
  check_FeOpComplete(diff16,bs,vector<unsigned int> {idx1,idx2,idx4,idx5});
  BOOST_TEST(bs.none());
  BOOST_TEST(diff16.n == 4);

  auto diff17 = -sum7;
  check_FeOpComplete(diff17,bs,vector<unsigned int> {idx1,idx2});
  BOOST_TEST(bs.none());
  BOOST_TEST(diff17.n == 4);

  auto diff18 = -sum8;
  check_FeOpComplete(diff18,bs,vector<unsigned int> {idx1,idx2});
  BOOST_TEST(bs.none());
  BOOST_TEST(diff18.n == 5);

  auto diff19 = -sum9;
  check_FeOpComplete(diff19,bs,vector<unsigned int> {idx1,idx2});
  BOOST_TEST(bs.none());
  BOOST_TEST(diff19.n == 5);

  auto diff20 = -sum10;
  check_FeOpComplete(diff20,bs,vector<unsigned int> {idx1,idx2,idx4,idx5});
  BOOST_TEST(bs.none());
  BOOST_TEST(diff20.n == 9);

  //something randomly complicated
  auto randomc = -(diff11 - sum9 + diff12 - sum8 + diff13 - sum7 - diff20 + sum5);
  check_FeOpComplete(randomc,bs,vector<unsigned int> {idx1,idx2,idx3,idx4,idx5});
  BOOST_TEST(bs.none());
  BOOST_TEST(randomc.n == 34);

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
#if 0
    sumfe_same_types<FEDivergence, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    sumfe_same_types<FESymmetricGradient, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    // sumfe_same_types<FECurl,obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index>(); 9. TBD
    // fails..no operator + for it. Analysis pending
    sumfe_same_types<FEGradient, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    sumfe_same_types<FELaplacian, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    sumfe_same_types<FEDiagonalHessian, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
    sumfe_same_types<FEHessian, obj_comb[i].rank, obj_comb[i].dim, obj_comb[i].index>();
#endif
  }
};
BOOST_FIXTURE_TEST_CASE(SumFEObjSameType, FEFixture)
{
  for_<0, 1>::run<SumFEfunctor>();

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
//      operation     output tensor rank
//        F         n
//        FG          n+1
//        FGD         n
//        FGG         n+2
//        FGGD        n+1
//        FD          n-1
// where e.g. FGD => div(grad(FEFunction F))
template <typename T>
void
sumfe_diff_types(const T& type1, const T& type2)
{
  auto sum1 = type1 + type2;
  auto sum2 = type2 + type1;
#if 0
  auto sum3 = sum1 + sum2;
  auto sum4 = sum1 + sum3;
#endif
}

BOOST_AUTO_TEST_CASE(SumFEObjDiffType)
{
  const int rank = 2;
  const int dim = 2;
  const int fe_no = 10; // random

  FEFunction<rank, dim, fe_no> F("test_obj");
  auto FG = grad(F);
  auto FGD = div(grad(F));
  // auto FGG = grad(grad(F)); TBD: Compilation fails.
  // auto FGGD = div(grad(grad(F))); TBD: Compilation fails.
  auto FD = div(F);

  sumfe_diff_types(F, F);
  sumfe_diff_types(FG, FG);
  sumfe_diff_types(FD, FD);
#if 0 // TBD: Compilation fail, due to missing definition.
  //sumfe_diff_types(F,FGD);
  //sumfe_diff_types(FG,FGGD);
  //sumfe_diff_types(FGD,FGD);
  //sumfe_diff_types(FGG,FGG);
  //sumfe_diff_types(FGGD,FGGD);
#endif
}
