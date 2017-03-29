///////////
// Test module for dealii_matrixfree.h
//////////
#define BOOST_TEST_MODULE TMOD_DEALII_MATRIXFREE_H
#define BOOST_TEST_DYN_LINK
#include <boost/test/included/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include "fe_data.h"
#include <deal.II/fe/fe_q.h>
#include <cfl/dealii_matrixfree.h>
#include <iostream>
#include <map>


using namespace std;
using namespace boost;
using namespace dealii;
using namespace CFL::dealii::MatrixFree;
namespace utf = boost::unit_test;

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


/////
typedef struct{
	int rank;
	int dim;
	unsigned int index;
	double scalar_factor;
}FEData_s;

typedef struct{
	int rank;
	int dim;
	unsigned int index;
	bool scalar_valued;
}TestData_s;
/////

//// Test case TestObjCreation
// Type: Positive test case
// Coverage: following classes - TestFunction, TestDivergence, TestSymmetricGradient, TestCurl, TestGradient, TestHessian
// Checks for:
// 1. Template instantiation with several combinations of (rank,dim,index) as per  variable obj_comb
// 2. Object creation in each case, and basic state check for the object
template<int i>
struct TestDatafunctor {
	static constexpr TestData_s obj_comb[9] = {
			// {rank, dim, index, scalar_valued}
	        {0,1,1,1}, {0,2,1,1}, {0,3,1,1},
	        {1,1,1,0}, {1,2,1,0}, {1,3,1,0},
	        {2,1,1,0}, {2,2,1,0}, {2,3,1,0}};
	typedef map<string, pair<bool,bool>> comb_map_t; //Testxxx class name, (integrate_value, integrate_gradient)
	static  comb_map_t i_comb;

	static void run() {
				//TBD: Change to avoid duplication??
        		TestFunction<obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index> test_obj1;
        		//Check basic state of object
        		BOOST_TEST(test_obj1.index == obj_comb[i].index);
        		BOOST_TEST(test_obj1.scalar_valued == obj_comb[i].scalar_valued);
        		BOOST_TEST(test_obj1.integrate_value == i_comb["TestFunction"].first);
        		BOOST_TEST(test_obj1.integrate_gradient == i_comb["TestFunction"].second);

        		TestDivergence<obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index> test_obj2;
        		//Check basic state of object
        		BOOST_TEST(test_obj2.index == obj_comb[i].index);
        		BOOST_TEST(test_obj2.scalar_valued == obj_comb[i].scalar_valued);
        		BOOST_TEST(test_obj2.integrate_value == i_comb["TestDivergence"].first);
        		BOOST_TEST(test_obj2.integrate_gradient == i_comb["TestDivergence"].second);

        		TestSymmetricGradient<obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index> test_obj3;
        		//Check basic state of object
        		BOOST_TEST(test_obj3.index == obj_comb[i].index);
        		BOOST_TEST(test_obj3.scalar_valued == obj_comb[i].scalar_valued);
        		BOOST_TEST(test_obj3.integrate_value == i_comb["TestSymmetricGradient"].first);
        		BOOST_TEST(test_obj3.integrate_gradient == i_comb["TestSymmetricGradient"].second);

        		TestCurl<obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index> test_obj4;
        		//Check basic state of object
        		BOOST_TEST(test_obj4.index == obj_comb[i].index);
        		BOOST_TEST(test_obj4.scalar_valued == obj_comb[i].scalar_valued);
        		BOOST_TEST(test_obj4.integrate_value == i_comb["TestCurl"].first);
        		BOOST_TEST(test_obj4.integrate_gradient == i_comb["TestCurl"].second);

        		TestGradient<obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index> test_obj5;
        		//Check basic state of object
        		BOOST_TEST(test_obj5.index == obj_comb[i].index);
        		BOOST_TEST(test_obj5.scalar_valued == obj_comb[i].scalar_valued);
        		BOOST_TEST(test_obj5.integrate_value == i_comb["TestGradient"].first);
        		BOOST_TEST(test_obj5.integrate_gradient == i_comb["TestGradient"].second);

        		TestHessian<obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index> test_obj6;
        		//Check basic state of object
        		BOOST_TEST(test_obj6.index == obj_comb[i].index);
        		BOOST_TEST(test_obj6.scalar_valued == obj_comb[i].scalar_valued);
        		BOOST_TEST(test_obj6.integrate_value == i_comb["TestHessian"].first);
        		BOOST_TEST(test_obj6.integrate_gradient == i_comb["TestHessian"].second);
        }
};

//ugly initialization of static member
template<int i> typename TestDatafunctor<i>::comb_map_t TestDatafunctor<i>::i_comb = {
		{"TestFunction", {true,false}},
		{"TestDivergence", {false,true}},
		{"TestSymmetricGradient", {false,true}},
		{"TestCurl", {false,true}},
		{"TestGradient", {false,true}},
		{"TestHessian", {false,false}}};

BOOST_AUTO_TEST_CASE(TestObjCreation) {
	//Check some combinations of Test objects
    for_<0, 3>::run<TestDatafunctor>();
}

////////////////////////////////////////////////////////////////////////////

//// Test case FEObjCreation
// Type: Positive test case
// Coverage: following classes - FEFunction, FEDivergence, FELiftDivergence, FESymmetricGradient,FECurl
//             					 FEGradient, FELaplacian, FEDiagonalHessian, FEHessain
//			 following functions - div(), grad()
// Checks for:
// 1. Template instantiation with several combinations of (rank,dim,index) as per  variable obj_comb
// 2. Object creation in each case, and basic state check for the object
template<int i>
struct FEfunctor {
	static constexpr FEData_s obj_comb[9] = {
			// {rank, dim, index, scalar-factor}
	        {0,1,1,0.1}, {0,2,1,0.2}, {0,3,1,0.3},
	        {1,1,1,0.4}, {1,2,1,0.5}, {1,3,1,0.6},
	        {2,1,1,0.7}, {2,2,1,0.8}, {2,3,1,0.9}};

	static void run() {
				//TBD: Change to avoid duplication??

				//Check possible constructions and basic state of objects
				FEFunction<obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index> test_fe_fun_obj1_1(); //default const
				FEFunction<obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index> test_fe_fun_obj1_2("test_fe_fun_obj1_2"); //one form of const
				FEFunction<obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index> test_fe_fun_obj1("test_fe_fun_obj1",obj_comb[i].scalar_factor); //second form of const
        		//Check basic state of object
        		BOOST_TEST(test_fe_fun_obj1.index == obj_comb[i].index);
        		BOOST_TEST(test_fe_fun_obj1.name() == "test_obj");
        		BOOST_TEST(test_fe_fun_obj1.scalar_factor == obj_comb[i].scalar_factor);

        		auto test_grad_obj = grad(test_fe_fun_obj1);
        		BOOST_TEST(test_grad_obj.index == test_fe_fun_obj1.index);
        		BOOST_TEST(test_grad_obj.name() == test_fe_fun_obj1.name());
        		BOOST_TEST(test_grad_obj.scalar_factor == test_fe_fun_obj1.scalar_factor);

#if 0  //TBD The construction fails due to no matching default constructor of base class. Discuss why the constructor
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

        		//TBD Add for other class after talk : see point 3
        }
};


BOOST_AUTO_TEST_CASE(FEObjCreation) {
	//Check some combinations of Test objects
    for_<0, 9>::run<FEfunctor>();
}


//// Test case FEObjMemFunc
// Type: Positive test case
// Coverage: following classes - FEFunction, FEDivergence, FELiftDivergence, FESymmetricGradient,FECurl
//             					 FEGradient, FELaplacian, FEDiagonalHessian, FEHessain
//			 following functions - div(), grad()
// Checks for:
// 1. Template instantiation of all member functions
//		Typically there is nothing more to test since the member functions ultimately call dealii library functions.
//		We only test state of our object and not dealii results
struct FEFixture{
	FEFixture():fedata_0_system(fe_shared),fedata_1_system(fe_shared),fedatas(fedata_0_system) {}
	~FEFixture() {};

	static const int fe_degree = 2;
	static const int n_components = 1;
	static const int dim = 2;
	static const int max_fe_degree = 2;
	static const int fe_no_0 = 10; //a nonzero random fe_no
	static const int fe_no_1 = 20; //a nonzero random fe_no
	using fe_q_type = FE_Q<dim>;
	using fe_shared_type = const std::shared_ptr<fe_q_type>;
	static fe_shared_type fe_shared;

	FEData<FE_Q, fe_degree, n_components, dim, fe_no_0, max_fe_degree, double> fedata_0_system;
	FEData<FE_Q, fe_degree, n_components, dim, fe_no_1, max_fe_degree, double> fedata_1_system;
	FEDatas<decltype(fedata_0_system)> fedatas;
};

FEFixture::fe_shared_type FEFixture::fe_shared = std::make_shared<FEFixture::fe_q_type>(2);

BOOST_FIXTURE_TEST_CASE(FEObjMemFunc,FEFixture,* utf::tolerance(0.00001)) {

	auto fedatas1 = (fedatas,fedata_1_system); //This is to invoke template hierarchy of FEDatas which are stored in FEFunction object

	FEFunction<0, dim, fe_no_0> fe_function_scalar("f_s");
	fe_function_scalar.value(fedatas,1);
	//fe_function_scalar.set_evaluation_flags(); possible Issue discuss..point 4, and add more cases for each class
	FEFunction<0, dim, fe_no_1> fe_function_scalar1("f_s");
	fe_function_scalar1.value(fedatas1,1);
	//TBD . behavior comments - see point 8
	auto fe_function_scalar2 = -fe_function_scalar;
	BOOST_TEST(fe_function_scalar2.scalar_factor == -fe_function_scalar.scalar_factor);
	auto fe_function_scalar3 = 0.2 * fe_function_scalar * 0.1;
	BOOST_TEST(fe_function_scalar3.scalar_factor == fe_function_scalar.scalar_factor*0.1*0.2);


	FEFunction<1, dim, fe_no_0> fe_function_vector("f_v");
	fe_function_vector.value(fedatas,1);
	FEFunction<1, dim, fe_no_1> fe_function_vector1("f_v");
	fe_function_vector1.value(fedatas1,1);
	auto fe_function_vector2 = -fe_function_vector;
	BOOST_TEST(fe_function_vector2.scalar_factor == -fe_function_vector.scalar_factor);
	auto fe_function_vector3 = 0.2 * fe_function_vector * 0.1;
	BOOST_TEST(fe_function_vector3.scalar_factor == fe_function_vector.scalar_factor*0.1*0.2);



	FEGradient<1, dim, fe_no_0> fe_gradient_scalar = grad(fe_function_scalar);
	fe_gradient_scalar.value(fedatas,1);
	FEGradient<1, dim, fe_no_1> fe_gradient_scalar1 = grad(fe_function_scalar1);
	fe_gradient_scalar1.value(fedatas1,1);
	auto fe_gradient_scalar2 = -fe_gradient_scalar;
	BOOST_TEST(fe_gradient_scalar2.scalar_factor == -fe_gradient_scalar.scalar_factor);
	auto fe_gradient_scalar3 = 0.2 * fe_gradient_scalar * 0.1;
	BOOST_TEST(fe_gradient_scalar3.scalar_factor == fe_gradient_scalar.scalar_factor*0.1*0.2);



	FEGradient<2, dim, fe_no_0> fe_gradient_vector = grad(fe_function_vector);
	fe_gradient_vector.value(fedatas,1);
	FEGradient<2, dim, fe_no_1> fe_gradient_vector1 = grad(fe_function_vector1);
	fe_gradient_vector1.value(fedatas1,1);
	auto fe_gradient_vector2 = -fe_gradient_vector;
	BOOST_TEST(fe_gradient_vector2.scalar_factor == -fe_gradient_vector.scalar_factor);
	auto fe_gradient_vector3 = 0.2 * fe_gradient_vector * 0.1;
	BOOST_TEST(fe_gradient_vector3.scalar_factor == fe_gradient_vector.scalar_factor*0.1*0.2);


	//TBD: fails in get_divergence missing definition, same as in fe_data.h
	//FEDivergence<0, dim, fe_no_0> fe_divergence = div(fe_function_vector);
	//fe_divergence.value(fedatas,1);

	//FELiftDivergence<decltype(fe_divergence)> lifted_divergence(fe_divergence); //TBD, yet to analyze

	//TBD: Fails in get_symmetric_gradient missing definition, same as in fe_data.h
	//FESymmetricGradient<1, dim, fe_no_0> fe_symmetric_gradient("f_s_g_s");
	//fe_symmetric_gradient.value(fedatas,1);

	//TBD: No definition of get_curl. Discuss, point no. 5
	//FECurl<1, dim, fe_no_0> fe_curl("f_c_s");
	//fe_curl.value(fedatas,1);

	FELaplacian<0, dim, fe_no_0> fe_laplacian = div(fe_gradient_scalar);
	fe_laplacian.value(fedatas,1);
	FELaplacian<0, dim, fe_no_1> fe_laplacian1 = div(fe_gradient_scalar1);
	fe_laplacian1.value(fedatas1,1);
	auto fe_laplacian2 = -fe_laplacian;
	BOOST_TEST(fe_laplacian2.scalar_factor == -fe_laplacian.scalar_factor);
	auto fe_laplacian3 = 0.2 * fe_laplacian * 0.1;
	BOOST_TEST(fe_laplacian3.scalar_factor == fe_laplacian.scalar_factor*0.1*0.2);


	FEDiagonalHessian<2, dim, fe_no_0> fe_diagonal_hessian("fe_d_h_s");
	fe_diagonal_hessian.value(fedatas,1);
	FEDiagonalHessian<2, dim, fe_no_1> fe_diagonal_hessian1("fe_d_h_s");
	fe_diagonal_hessian1.value(fedatas1,1);
	auto fe_diagonal_hessian2 = -fe_diagonal_hessian;
	BOOST_TEST(fe_diagonal_hessian2.scalar_factor == -fe_diagonal_hessian.scalar_factor);
	auto fe_diagonal_hessian3 = 0.2 * fe_diagonal_hessian * 0.1;
	BOOST_TEST(fe_diagonal_hessian3.scalar_factor == fe_diagonal_hessian.scalar_factor*0.1*0.2);


	FEHessian<2, dim, fe_no_0> fe_hessian("f_h_s");
	fe_hessian.value(fedatas,1);
	FEHessian<2, dim, fe_no_1> fe_hessian1("f_h_s");
	fe_hessian1.value(fedatas1,1);
	auto fe_hessian2 = -fe_hessian;
	BOOST_TEST(fe_hessian2.scalar_factor == -fe_hessian.scalar_factor);
	auto fe_hessian3 = 0.2 * fe_hessian * 0.1;
	BOOST_TEST(fe_hessian3.scalar_factor == fe_hessian.scalar_factor*0.1*0.2);
}


//// Test case SumFEObjSameType
// Type: Positive test case
// Coverage: following classes - SumFEFunctions
// Checks for:
// 1. Behavior for overloaded operator "+" when types of operands is exactly same
//		This is done by "invoking" operator overloading for several combinations which can conclude that:
// 		- all possible ways to sum FE functions is tested
//		- result of SumFEFunction is of correct type
//		There are no check on values since they are available from dealii library function results which we ignore in out test
template <template <int,int,unsigned int> typename FEFuncType,int rank, int dim, unsigned int idx>
void sumfe_same_types() {
	using FEFunc = FEFuncType<rank, dim, idx>;

	FEFunc fe_function1("test_fe1");
	FEFunc fe_function2("test_fe2");
	FEFunc fe_function3("test_fe3");

	auto sum1 = fe_function1 + fe_function2;
	auto sum2 = fe_function2 + fe_function1;
	auto sum3 = fe_function2 + fe_function1 + fe_function3;
	auto sum4 = sum1 + fe_function1 + fe_function2;
	auto sum5 = fe_function1 + fe_function2 + sum1;
	auto sum6 = sum1 + sum2;
	auto sum7 = sum1 + sum2 + fe_function1;
	auto sum8 = fe_function2 + sum1 + sum2;
	auto sum9 = sum6 + sum8;
}

template<int i>
struct SumFEfunctor {
	static constexpr FEData_s obj_comb[9] = {
			// {rank, dim, index, scalar-factor}
	        {0,1,1,0.1}, {0,2,1,0.2}, {0,3,1,0.3},
	        {1,1,1,0.4}, {1,2,1,0.5}, {1,3,1,0.6},
	        {2,1,1,0.7}, {2,2,1,0.8}, {2,3,1,0.9}};

	static void run() {
		sumfe_same_types<FEFunction,obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index>();
		sumfe_same_types<FEDivergence,obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index>();
		sumfe_same_types<FESymmetricGradient,obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index>();
		//sumfe_same_types<FECurl,obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index>(); 9. TBD fails..no operator + for it. Analysis pending
		sumfe_same_types<FEGradient,obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index>();
		sumfe_same_types<FELaplacian,obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index>();
		sumfe_same_types<FEDiagonalHessian,obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index>();
		sumfe_same_types<FEHessian,obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index>();
	}
};
BOOST_FIXTURE_TEST_CASE(SumFEObjSameType,FEFixture)
{
	for_<0, 9>::run<SumFEfunctor>();

	//auto sumf = fe_function_scalar1 + fe_function_scalar2; //TBD why does this not hit static_assertion although compilation fails?
}


//// Test case SumFEObjDiffType
// Type: Positive test case
// Coverage: following classes - SumFEFunctions
// Checks for:
// 1. Behavior for overloaded operator "+" when types of operands is different but tensor rank is equal
//		This is done by "invoking" operator overloading for several combinations which can conclude that:
// 		- all possible ways to sum FE functions is tested
//		- result of SumFEFunction is of correct type
//		There are no check on values since they are available from dealii library function results which we ignore in out test
// Remarks:
// 1. How are the combinations of different types chosen?
//		See that grad() returns FEFunction of rank+1, while div returns FEFunction of rank-1
// 		The state machine would be (< and > should be seen as arrow termination):
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
//		i.e. valid state transitions are
// 			operation			output tensor rank
//				F					n
//				FG					n+1
//				FGD					n
//				FGG					n+2
//				FGGD				n+1
//				FD					n-1
// where e.g. FGD => div(grad(FEFunction F))
template <typename T>
void sumfe_diff_types(const T& type1, const T& type2)
{
	auto sum1 = type1 + type2;
	auto sum2 = type2 + type1;
	auto sum3 = sum1 + sum2;
	auto sum4 = sum1 + sum3;
}

BOOST_AUTO_TEST_CASE(SumFEObjDiffType){
	const int rank = 2;
	const int dim = 2;
	const int fe_no = 10; //random

	FEFunction<rank, dim, fe_no> F("test_obj");
	auto FG = grad(F);
	auto FGD = div(grad(F));
	//auto FGG = grad(grad(F)); TBD: Compilation fails. Analysis pending see 13
	//auto FGGD = div(grad(grad(F))); TBD: Compilation fails. Analysis pending see 13
	auto FD = div(F);

	sumfe_diff_types(F,F);
	sumfe_diff_types(FG,FG);
	sumfe_diff_types(FD,FD);
#if 0 //TBD: Compilation fail, due to missing definition. see point no. 10
	//sumfe_diff_types(F,FGD);
	//sumfe_diff_types(FG,FGGD);
	//sumfe_diff_types(FGD,FGD);
	//sumfe_diff_types(FGG,FGG);
	//sumfe_diff_types(FGGD,FGGD);
#endif
}

//// Test case ProdFEObjSameType
// Type: Positive test case
// Coverage: following classes - ProductFEFunctions
// Checks for:
// 1. Behavior for overloaded operator "*" when types of operands is exactly same
//		This is done by "invoking" operator overloading for several combinations which can conclude that:
// 		- all possible ways to multiply FE functions is tested
//		- result of ProductFEFunction is of correct type
//		There are no check on values since they are available from dealii library function results which we ignore in out test
template <template <int,int,unsigned int> typename FEFuncType,int rank, int dim, unsigned int idx>
void prodfe_same_types() {
	using FEFunc = FEFuncType<rank, dim, idx>;

	FEFunc fe_function1("test_fe1");
	FEFunc fe_function2("test_fe2");
	FEFunc fe_function3("test_fe3");

	auto prod1 = fe_function1 * fe_function2;
	auto prod2 = fe_function2 * fe_function1;
	auto prod3 = fe_function2 * fe_function1 * fe_function3;
	auto prod4 = prod1 * fe_function1 * fe_function2;
	auto prod5 = fe_function1 * fe_function2 * prod1;
	auto prod6 = prod1 * prod2;
	auto prod7 = prod1 * prod2 * fe_function1;
	auto prod8 = fe_function2 * prod1 * prod2;
	auto prod9 = prod6 * prod8;
}

template<int i>
struct ProdFEfunctor {
	static constexpr FEData_s obj_comb[] = {
			// {rank, dim, index, scalar-factor}
	        {0,1,1,0.1}, {0,2,1,0.2}, {0,3,1,0.3},
	        {1,1,1,0.4}, {1,2,1,0.5}, {1,3,1,0.6},
	        {2,1,1,0.7}, {2,2,1,0.8}, {2,3,1,0.9}};

	static void run() {
		prodfe_same_types<FEFunction,obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index>();
		prodfe_same_types<FEDivergence,obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index>();
		prodfe_same_types<FESymmetricGradient,obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index>();
		//prodfe_same_types<FECurl,obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index>(); 12. TBD fails..no operator * for it. Analysis pending
		prodfe_same_types<FEGradient,obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index>();
		prodfe_same_types<FELaplacian,obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index>();
		prodfe_same_types<FEDiagonalHessian,obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index>();
		prodfe_same_types<FEHessian,obj_comb[i].rank,obj_comb[i].dim,obj_comb[i].index>();
	}
};
BOOST_FIXTURE_TEST_CASE(ProdFEObjSameType,FEFixture)
{
	for_<0, 9>::run<ProdFEfunctor>();
}

//// Test case ProdFEObjDiffType
// Type: Positive test case
// Coverage: following classes - ProductFEFunctions
// Checks for:
// 1. Behavior for overloaded operator "*" when types of operands is different but tensor rank is equal
//		This is done by "invoking" operator overloading for several combinations which can conclude that:
// 		- all possible ways to multiply FE functions is tested
//		- result of ProductFEFunction is of correct type
//		There are no check on values since they are available from dealii library function results which we ignore in out test
// Remarks:
// See remark 1 from SumFEObjDiffType
template <typename T>
void prodfe_diff_types(const T& type1, const T& type2)
{
	auto prod1 = type1 * type2;
	auto prod2 = type2 * type1;
	auto prod3 = prod1 * prod2;
	auto prod4 = prod1 * prod3;
}

BOOST_AUTO_TEST_CASE(ProdFEObjDiffType){
	const int rank = 2;
	const int dim = 2;
	const int fe_no = 10; //random

	FEFunction<rank, dim, fe_no> F("test_obj");
	auto FG = grad(F);
	auto FGD = div(grad(F));
	//auto FGG = grad(grad(F)); TBD: Compilation fails. Analysis pending see 13
	//auto FGGD = div(grad(grad(F))); TBD: Compilation fails. Analysis pending see 13
	auto FD = div(F);

	prodfe_diff_types(F,F);
	prodfe_diff_types(FG,FG);
	prodfe_diff_types(FD,FD);
 //TBD: Compilation fail, due to missing definition. see point no. 10
#if 0
	//prodfe_diff_types(F,FGD);
	//prodfe_diff_types(FG,FGGD);
	//prodfe_diff_types(FGD,FGD);
	//prodfe_diff_types(FGG,FGG);
	//prodfe_diff_types(FGGD,FGGD);
#endif
}

//TBD. Questions
//1. The construction of FEGradient and FEHessian (atleast) is ambiguous, since we allow construction with or without the
//   function/gradient. i.e. we can create such an object with just a string or with function/gradient. Why so?
//2. The construction of Hessian, Laplacian, Divergence etc do not store the state from the object with which they were created. Why?
//3. In general, why there are no helper functions for FELiftDivergencem FESymmetricGradient, FECurl, FEDiagonalHessian, and should they be added?
//4. There seems to be no way to get to call set_evaluation_flags() function. This needs FEEvaluation object which is not accessible from FEDatas.
//   I wonder if we instead wanted to call the similar function of FEDatas?
//5. No definition of get_curl, discuss
//6. jfi..fixed an issue in line 714 in dealii_matrixfree.h
//7. Remark: I think scalar_factor should be made private with a getter function
//8. Behavior points
//  Following are not possible (due to impossible default assignment operator=). Is this ok from user pov?
//  fe_function_scalar = -fe_function_scalar;
//  fe_function_scalarxx = fe_function_scalar - 10; // not defined operator - in this way
//  fe_function_scalar = fe_function_scalar * 2;
//9. No matching + operator for FECurl. Also, how do we test the result..right now its implicitly verified that the return type is also SumFEFunction
//10. Theoretically, these combinations are possible. Should these be fixed? Both in sum and product
//11. ProductFEFunctions constructor - the static_assertion text should say Product instead of add
//12. No matching * operator for FECurl. Also, how do we test the result..right now its implicitly verified that the return type is also ProdFEFunction
//13. Hessian and div(hessian) fail compilation test
//14. It is currently possible to create a div object initialized with scalar FEFunction. cant check value() due to another error. But should not this be
// refused at compilation time?
