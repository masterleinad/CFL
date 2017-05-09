#ifndef TEST_MATRIXFREE_H_
#define TEST_MATRIXFREE_H_

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include "test_utils.h"

#include <cfl/dealii_matrixfree.h>

#include <deal.II/fe/fe_q.h>
#include <dealii/fe_data.h>

#include <iostream>
#include <map>
#include <bitset>

using namespace std;
using namespace boost;
using namespace dealii;
using namespace CFL::dealii::MatrixFree;
namespace utf = boost::unit_test;

/////
typedef struct
{
  int rank;
  int dim;
  unsigned int index;
  double scalar_factor;
} FEFunction_s;

typedef struct
{
  int rank;
  int dim;
  unsigned int index;
  bool scalar_valued;
} TestData_s;
/////

struct FEFixture
{
  FEFixture()
    : fedata_0_system(fe_shared)
    , fedata_1_system(fe_shared)
    , fedatas(fedata_0_system)
  {
  }
  ~FEFixture(){};

  static const int fe_degree = 2;
  static const int n_components = 1;
  static const int dim = 2;
  static const int max_fe_degree = 2;
  static const int fe_no_0 = 10; // a nonzero random fe_no
  static const int fe_no_1 = 20; // a nonzero random fe_no
  using fe_q_type = FE_Q<dim>;
  using fe_shared_type = const std::shared_ptr<fe_q_type>;
  static fe_shared_type fe_shared;

  FEData<FE_Q, fe_degree, n_components, dim, fe_no_0, max_fe_degree, double> fedata_0_system;
  FEData<FE_Q, fe_degree, n_components, dim, fe_no_1, max_fe_degree, double> fedata_1_system;
  FEDatas<decltype(fedata_0_system)> fedatas;
};

FEFixture::fe_shared_type FEFixture::fe_shared = std::make_shared<FEFixture::fe_q_type>(2);

template <typename FEOp, typename BitSetType>
void
check_FeOpComplete(const FEOp& feopobj, BitSetType& bs, vector<unsigned int>&& v)
{
  bs.reset();
  for (unsigned int i : v)
    bs[i - 1] = 1; // set the corresponding bits
  // cout<<"initial bits "<<bs<<endl;
  for (unsigned int i = 1; i <= feopobj.n; i++)
    bs[feopobj.get_fe_func_index(i) - 1] = 0; // reset as and when found the FEFunc
};

#endif /* TEST_MATRIXFREE_H_ */
