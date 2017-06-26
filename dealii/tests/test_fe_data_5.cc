//////////
#define BOOST_TEST_MODULE TMOD_FEDATA_5_H
#define BOOST_TEST_DYN_LINK
#include "test_fe_data.h"
//////////

//// Test case FEDatasMemFunc

//Remark: This test case is not abandoned..since until the construction of
// of a proper MatrixFree object, we cant check the behavor of internal functions
// The construction of MatrixFree is very problem specific (to my understanding),
// and we dont intend to solve a complete problem during UT!

//  Positive test cases for FEDatas class, checks .
// Type: Positive test case
// Coverage: following classes - FEDatas
// Checks for:
// 1. template instantiation of all member functions, way down to deepest inheritance hierarchy
//    It does not checks the outcome since they are eventual calls to dealii library
// functions, and we only test
//    state of our object in UT
template <int i>
struct FEDatasfunctor
{
  static void
  run()
  {
    FEDatasFixture fixtureObj;
    auto fedatas = (fixtureObj.fedata_0_system,
                    fixtureObj.fedata_1_system,
                    fixtureObj.fedata_2_system,
                    fixtureObj.fedata_3_system,
                    fixtureObj.fedata_4_system);

#if 0
    //TBD: This fails during construction of FEEvaluation object. Discuss
    const int dim = 2;
    MatrixFree<dim> data;
    fedatas.initialize<dim>(data);
#endif

    fedatas.rank<i>();

    // fedatas.evaluate(); //Fails due to missing initialization - FIXIT
    fedatas.integrate();
    fedatas.set_integration_flags<i>(true, true);
    fedatas.set_evaluation_flags<i>(true, true, true);

    fedatas.get_n_q_points<i>();
#if 0 // FIXIT: Gives fatal error: in "FEDatasMemFunc": memory access violation at address:
    // 0x000000be: no mapping at fault address
    //fedatas.get_gradient<i>(0);
    //fedatas.get_symmetric_gradient<i>(0); //TBD Missing definition error, discuss
    //fedatas.get_divergence<i>(0); //TBD Missing definition error, discuss
    fedatas.get_laplacian<i>(0);
    fedatas.get_hessian_diagonal<i>(0);
    fedatas.get_hessian<i>(0);
    fedatas.get_value<i>(0);
    fedatas.dofs_per_cell<i>();
    fedatas.begin_dof_values<i>();
#endif
    fedatas.tensor_dofs_per_cell<i>();

#if 0 // TBD to test
    reinit<Cell>
    read_dof_values<VectorType>
    distribute_local_to_global<VectorType>
    submit_curl<VectorType,q)
    submit_divergence<VectorType,fe_no>
    submit_symmetric_gradient<VectorType,fe_no>)
    submit_gradient<VectorType,fe_no>)
    submit_value<VectorType,fe_no>)
#endif
  }
};

BOOST_AUTO_TEST_CASE(FEDatasMemFunc)
{
  // Check some combinations of object creation
  for_<0, 5>::run<FEDatasfunctor>();
}
