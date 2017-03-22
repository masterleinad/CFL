///////////
// Test module for fe_data.h
//////////
#define BOOST_TEST_MODULE TMOD_FEDATA_H
#define BOOST_TEST_DYN_LINK
#include <boost/test/included/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>


#include "fe_data.h"
#include <deal.II/fe/fe_q.h>
#include <iostream>

using namespace dealii;
using namespace std;
namespace bdata = boost::unit_test::data;

///////////// General definition for a static for loop required for instantiation of  templates/////////
template<int from, int to>
struct for_ {
        template<template<int> class Fn>
        static void run() {
                Fn<from>::run();
                for_<from + 1, to>::template run<Fn>();
        }
};

template<int to>
struct for_<to, to> {
        template<template<int> class Fn>
        static void run() {
        }
};
/////////////////////////////////////////////////////////////////////////////////////




//// Test case FEDataCreation
// Type: Positive test case
// Coverage: following classes - FEData
// Checks for:
// 1. Template instantiation with several combinations of (degree,number of components,dim,index)
// 2. Object creation in each case, and basic state check for the object
template<int i>
struct FEDatafunctor {
	static constexpr int degree[] = { 1, 2, 3 };
	static constexpr int n_comp[] = { 1, 1, 1 }; //todo, change n_comp by using FESystem
	static constexpr int dim[] = { 1, 2, 3 };
	static constexpr int fe_no[] = { 0, 1, 2 };
	static constexpr int max_degree = 4;

        static void run() {
        		FE_Q<dim[i]> fe(degree[i]);

        		FEData<FE_Q,degree[i],n_comp[i],dim[i],fe_no[i],max_degree> obj(fe);

        		//Check basic state of object
        		BOOST_TEST(obj.fe_number == fe_no[i]);
        		BOOST_TEST(obj.max_degree == max_degree);
        }
};

BOOST_AUTO_TEST_CASE(FEDataCreation) {
	//Check some combinations of object creation
    for_<0, 3>::run<FEDatafunctor>();
}


//// Test case FEDataPositive
// Type: Positive test case
// Coverage: following classes - FEData
// Checks for:
// 1. Object construction
// 2. behavior of overloaded comma operator in this class
// 3. basic state check for the object in each case
BOOST_AUTO_TEST_CASE(FEDataPositive) {

		//Check that all shared pointers point to a common managed object
        const auto fe_shared = std::make_shared<FE_Q<2>>(2);
        BOOST_TEST(fe_shared.use_count() == 1);
        FEData<FE_Q, 2, 1, 2, 0, 2, double> fedata_e_system(fe_shared);
        BOOST_TEST(fe_shared.use_count() == 2);
        FEData<FE_Q, 2, 1, 2, 1, 2, double> fedata_u_system(fe_shared);
        BOOST_TEST(fe_shared.use_count() == 3);
        FEData<FE_Q, 2, 1, 2, 2, 2, double> fedata_x_system(fe_shared);
        BOOST_TEST(fe_shared.use_count() == 4);
        
        //Check the state of objects added through comma operator
        //fedata_e_system,fedata_u_system; //ok..but this is strange, maybe it can just be ignored
        auto fedatas_2 = (fedata_e_system,fedata_u_system); //ok, but using () has to be remembered
        BOOST_TEST(fedatas_2.get_fe_data<0>().fe_number == 0);
        BOOST_TEST(fedatas_2.get_fe_data<1>().fe_number == 1);
        auto fedatas_3 = (fedata_e_system,fedata_u_system,fedata_x_system); //ok, but using () has to be remembered
        BOOST_TEST(fedatas_3.get_fe_data<0>().fe_number == 0);
        BOOST_TEST(fedatas_3.get_fe_data<1>().fe_number == 1);
        BOOST_TEST(fedatas_3.get_fe_data<2>().fe_number == 2);
        
}


//// Test case FEDataNegative
// Type: Negative test case
// Coverage: following classes - FEData
// Checks for:
//	1. Runtime exceptions
BOOST_AUTO_TEST_CASE(FEDataNegative) {
	//Mismatch in FE degree
	typedef FEData<FE_Q,3/*unequal degree*/,1,2,1,4> ErrFEData1;
	FE_Q<2> fe(2);
	BOOST_CHECK_THROW(ErrFEData1 obj(fe),dealii::ExcIndexRange );

	//Mmismatch in number of components
	typedef FEData<FE_Q,2,2/*unequal components*/,2,1,4> ErrFEData2;
	BOOST_CHECK_THROW(ErrFEData2 obj(fe),dealii::ExcDimensionMismatch );
}



//// Test case FEDatasComma
// Type: Positive test case
// Coverage: following classes - FEDatas
// Checks for:
// 1. Object construction
// 2. behavior of overloaded comma operator in this class
//		This is done by "invoking" operator overloading for several combinations which can conclude that:
// 		- all possible ways to adding FEData objects in FEDatas static container is tested
// 3. basic state check for the object in each case
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


BOOST_FIXTURE_TEST_CASE(FEDatasComma,FEDatasFixture) {

        FEDatas<decltype(fedata_0_system)> fedatas(fedata_0_system);
        BOOST_TEST(fedatas.fe_number == fedatas.get_fe_data<0>().fe_number);

        auto fedatas1((fedata_0_system,fedata_1_system)); //invoking automatic copy construction
        BOOST_TEST(fedatas1.get_fe_data<0>().fe_number == 0);
        BOOST_TEST(fedatas1.get_fe_data<1>().fe_number == 1);
        
        //fedatas = (fedatas,fedata_1_system); TBD, cant be supported see point no. 9

        //Check the state of objects added through comma operator        
        FEDatas<decltype(fedata_0_system)> fedatas2 = fedata_0_system; //fails due to explicit constructor disallowing copy initialization

        auto fedatas3 = (fedatas2,fedata_1_system);
        BOOST_TEST(fedatas3.get_fe_data<0>().fe_number == 0);
        BOOST_TEST(fedatas3.get_fe_data<1>().fe_number == 1);
        
        auto fedatas4 = (fedatas,fedata_1_system,fedata_2_system);
        BOOST_TEST(fedatas4.get_fe_data<0>().fe_number == 0);
        BOOST_TEST(fedatas4.get_fe_data<1>().fe_number == 1);
        BOOST_TEST(fedatas4.get_fe_data<2>().fe_number == 2);

        
        auto fedatas5 = (fedata_1_system,fedatas);
        BOOST_TEST(fedatas5.get_fe_data<0>().fe_number == 0);
        BOOST_TEST(fedatas5.get_fe_data<1>().fe_number == 1);

        auto fedatas6 = (fedatas,fedata_1_system,fedata_2_system);
        BOOST_TEST(fedatas6.get_fe_data<0>().fe_number == 0);
        BOOST_TEST(fedatas6.get_fe_data<1>().fe_number == 1);
        BOOST_TEST(fedatas6.get_fe_data<2>().fe_number == 2);

        //TBD: This fails since we cant combine FEDatas with comma operator. Discuss if needed
#if 0
        auto fedatas_t1 = (fedatas,fedata_0_system,fedata_1_system);
        auto fedatas_t2 = (fedata_2_system,fedata_3_system);
        auto fedatas6 = (fedatas_t1,fedatas_t2);
        BOOST_TEST(fedatas6.get_fe_data<0>().fe_number == 0);
        BOOST_TEST(fedatas6.get_fe_data<1>().fe_number == 1);
#endif


#if 0
        MatrixFree<dim> data;
        fedatas.initialize<dim>(data);

        //TBD: Add tests for initialize of more fedatas'n' objects
#endif

}

//// Test case FEDatasMemFunc
//  Positive test cases for FEDatas class, checks .
// Type: Positive test case
// Coverage: following classes - FEDatas
// Checks for:
// 1. template instantiation of all member functions, way down to deepest inheritance hierarchy
// 		It does not checks the outcome since they are eventual calls to dealii library functions, and we only test
//		state of our object in UT
template<int i>
struct FEDatasfunctor {
        static void run() {
        		FEDatasFixture fixtureObj;
        		auto fedatas = (fixtureObj.fedata_0_system,fixtureObj.fedata_1_system,fixtureObj.fedata_2_system,
        						fixtureObj.fedata_3_system,fixtureObj.fedata_4_system);

#if 0
        		//TBD: This fails during construction of FEEvaluation object. Discuss
        		const int dim = 2;
        		MatrixFree<dim> data;
        		fedatas.initialize<dim>(data);
#endif

        		fedatas.rank<i>();

        		//fedatas.evaluate(); //Fails due to missing initialization - FIXIT
        		fedatas.integrate();
        		fedatas.set_integration_flags<i>(true,true);
        		fedatas.set_evaluation_flags<i>(true,true,true);

        		fedatas.get_n_q_points<i>();
#if 0 //FIXIT: Gives fatal error: in "FEDatasMemFunc": memory access violation at address: 0x000000be: no mapping at fault address
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

#if 0 //TBD to test
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


BOOST_AUTO_TEST_CASE(FEDatasMemFunc) {
	//Check some combinations of object creation
	for_<0, 5>::run<FEDatasfunctor>();

}


// Questions/other TBD
// 0. Discuss the issues found above
// 1. fe_no should be checked for uniqueness before adding
// 2. No way to remove an FEData pushed to FEDatas - is it not needed?
// 3. No mechanism to count the number of FEData objects in the list - is it not needed?
// 4. Comma operator overloading - just a discussion - X
// 5. Why not combine initialization and construction of FEDatas to prevent null pointer exceptions?
// 6. If not 5, Assert should be added to every member function of FEDatas
// 7. Add documentation for each class
// 8. Boost test lib and include files are not added by default in dealii installation. Can we get it added?
// 9. Such an expression can not be supported by variadic template. Is it ok from user pov?
