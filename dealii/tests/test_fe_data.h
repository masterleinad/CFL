#ifndef _TEST_FE_DATA_H_
#define _TEST_FE_DATA_H_

#include <boost/test/unit_test.hpp>
#include <dealii/fe_data.h>
#include <deal.II/fe/fe_q.h>
#include <iostream>
#include "test_utils.h"

using namespace dealii;
using namespace std;


//Just to avoid writing a U (unsigned) during comparison and prevent warning
const unsigned int fe_0 = 0, fe_1 = 1, fe_2 = 2, fe_3 = 3, fe_4 = 4;
///////////////////

struct FEDatasFixture{
	//This fixture provides 5 objects of FEData, each with a unique fe_number
	FEDatasFixture():fedata_0_system(fe_shared),fedata_1_system(fe_shared),fedata_2_system(fe_shared),
					 fedata_3_system(fe_shared),fedata_4_system(fe_shared) {}
	~FEDatasFixture() {}

	static const int fe_degree = 2;
	static const int n_components = 1;
	static const int dim = 2;
	static const int max_fe_degree = 2;
	using fe_q_type = FE_Q<dim>;
	using fe_shared_type = const std::shared_ptr<fe_q_type>;
	static fe_shared_type fe_shared;

	FEData<FE_Q, fe_degree, n_components, dim, 0, max_fe_degree, double> fedata_0_system;
	FEData<FE_Q, fe_degree, n_components, dim, 1, max_fe_degree, double> fedata_1_system;
	FEData<FE_Q, fe_degree, n_components, dim, 2, max_fe_degree, double> fedata_2_system;
	FEData<FE_Q, fe_degree, n_components, dim, 3, max_fe_degree, double> fedata_3_system;
	FEData<FE_Q, fe_degree, n_components, dim, 4, max_fe_degree, double> fedata_4_system;
};

FEDatasFixture::fe_shared_type FEDatasFixture::fe_shared = std::make_shared<FEDatasFixture::fe_q_type>(2);



#endif
